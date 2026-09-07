import os
os.environ["HF_HUB_OFFLINE"] = "1"
import pickle
import pandas as pd
from bs4 import BeautifulSoup
import pyterrier as pt

from src.text_processing import clean_wikitext, preprocess
import src.autocomplete as autocomplete

DATA_XML_PATH = "data/simplewiki-latest-pages-articles.xml"
MAX_ARTICLES = 1400  
INDEX_DIR = os.path.abspath("index/pyterrier_index")
DOCSTORE_PATH = "index/docstore.pkl"
BERT_EMB_PATH = "index/bert_embeddings.pkl"
ELMO_EMB_PATH = "index/elmo_embeddings.pkl"
BUILD_ELMO = True
BUILD_AUTOCOMPLETE = True


def load_articles(xml_path, max_articles):
    with open(xml_path, "r", encoding="utf-8") as file:
        content = file.read()
    parsed = BeautifulSoup(content, "xml")
    pages = parsed.find_all("page")
    articles = []
    for page in pages[:max_articles]:
        title = page.find("title").text
        article = page.find("text")
        if article:
            raw_text = article.text
        else:
            raw_text = ""
        articles.append({"title": title, "raw_text": raw_text})
    return articles


def build_dataframe():
    articles = load_articles(DATA_XML_PATH, MAX_ARTICLES)
    df = pd.DataFrame(articles)
    before = len(df)
    is_redirect = df["raw_text"].str.strip().str.lower().str.startswith("#redirect")
    df = df[~is_redirect].reset_index(drop=True)
    if before - len(df):
        print(f"dropped {before - len(df)} redirect page(s)")

    df["text"] = df["raw_text"].apply(clean_wikitext)
    df["processed_text"] = df["text"].apply(preprocess)
    before = len(df)
    df = df[df["processed_text"].str.strip() != ""].reset_index(drop=True)
    dropped = before - len(df)
    if dropped:
        print(f"dropped {dropped} empty article(s) after cleaning/preprocessing")

    df["docno"] = df.index.astype(str)
    return df[["docno", "title", "text", "processed_text"]]


def build_pyterrier_index(df):
    if not pt.java.started():
        pt.init(boot_packages=["com.github.terrierteam:terrier-prf:-SNAPSHOT"])
    index_df = df[["docno", "processed_text"]].rename(columns={"processed_text": "text"})
    indexer = pt.IterDictIndexer(INDEX_DIR, overwrite=True)
    index_ref = indexer.index(index_df.to_dict(orient="records"))
    return index_ref


def save_docstore(df):
    docstore = {}
    for row in df.itertuples():
        docstore[row.docno] = {"title": row.title, "text": row.text}
    with open(DOCSTORE_PATH, "wb") as file:
        pickle.dump(docstore, file)


def build_bert_embeddings(df):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(df["text"].tolist(), convert_to_numpy=True)
    embedding_map = {}
    for docno, embedding in zip(df["docno"], embeddings):
        embedding_map[docno] = embedding
    with open(BERT_EMB_PATH, "wb") as file:
        pickle.dump(embedding_map, file)


def build_elmo_embeddings(df):
    import tensorflow as tf
    import tensorflow_hub as hub
    import time
    embedding_map = {}
    if os.path.exists(ELMO_EMB_PATH):
        with open(ELMO_EMB_PATH, "rb") as file:
            embedding_map = pickle.load(file)
        print(f"resuming: {len(embedding_map)} document(s) already done from a previous run")

    elmo_model = hub.load("https://tfhub.dev/google/elmo/3")
    total = len(df)
    start = time.time()
    done_this_run = 0

    for docno, text in zip(df["docno"], df["text"]):
        if docno in embedding_map:
            continue

        embedding = elmo_model.signatures["default"](tf.constant([text]))["elmo"][0]
        embedding_map[docno] = embedding.numpy().mean(axis=0)
        done_this_run += 1

        if done_this_run % 25 == 0:
            elapsed = time.time() - start
            print(f"  {len(embedding_map)}/{total} done ({elapsed:.0f}s elapsed this run)")
            with open(ELMO_EMB_PATH, "wb") as file:
                pickle.dump(embedding_map, file)

    with open(ELMO_EMB_PATH, "wb") as file:
        pickle.dump(embedding_map, file)
    print(f"elmo embeddings complete: {len(embedding_map)}/{total}")


def main():
    os.makedirs(INDEX_DIR, exist_ok=True)
    df = build_dataframe()
    build_pyterrier_index(df)
    save_docstore(df)
    build_bert_embeddings(df)
    if BUILD_ELMO:
        build_elmo_embeddings(df)
    if BUILD_AUTOCOMPLETE:
        autocomplete.train_and_save_rnn(df)


if __name__ == "__main__":
    main()