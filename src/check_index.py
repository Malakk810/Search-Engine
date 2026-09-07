import pickle
import numpy as np
import pyterrier as pt

import src.search_service as ss

DOCSTORE_PATH = "index/docstore.pkl"
BERT_EMB_PATH = "index/bert_embeddings.pkl"
ELMO_EMB_PATH = "index/elmo_embeddings.pkl"


def check_counts():
    print("=== document counts ===")
    with open(DOCSTORE_PATH, "rb") as f:
        docstore = pickle.load(f)
    with open(BERT_EMB_PATH, "rb") as f:
        bert_emb = pickle.load(f)
    try:
        with open(ELMO_EMB_PATH, "rb") as f:
            elmo_emb = pickle.load(f)
    except FileNotFoundError:
        elmo_emb = {}

    if not pt.java.started():
        pt.init()
    index = pt.IndexFactory.of(ss.INDEX_DIR)
    index_doc_count = index.getCollectionStatistics().getNumberOfDocuments()

    print(f"docstore entries:      {len(docstore)}")
    print(f"bert embeddings:       {len(bert_emb)}")
    print(f"elmo embeddings:       {len(elmo_emb)}")
    print(f"pyterrier index docs:  {index_doc_count}")

    counts = {len(docstore), len(bert_emb), index_doc_count}
    if elmo_emb:
        counts.add(len(elmo_emb))
    if len(counts) == 1:
        print("all counts match")
    else:
        print("MISMATCH -- these should all be equal")

    return docstore, bert_emb, elmo_emb


def check_nans(bert_emb, elmo_emb):
    print("\n=== checking for NaN embeddings ===")
    bert_nan_docs = [docno for docno, vec in bert_emb.items() if np.isnan(vec).any()]
    print(f"bert vectors with NaN: {len(bert_nan_docs)}")
    if bert_nan_docs:
        print("  docnos:", bert_nan_docs[:10])

    if elmo_emb:
        elmo_nan_docs = [docno for docno, vec in elmo_emb.items() if np.isnan(vec).any()]
        print(f"elmo vectors with NaN: {len(elmo_nan_docs)}")
        if elmo_nan_docs:
            print("  docnos:", elmo_nan_docs[:10])


def check_sample_content(docstore, n=3):
    print(f"\n=== sample of {n} documents from docstore ===")
    for docno in list(docstore.keys())[:n]:
        doc = docstore[docno]
        preview = doc["text"][:200].replace("\n", " ")
        print(f"\ndocno {docno} -- {doc['title']}")
        print(f"  {preview}...")


def check_search(query, models):
    print(f"\n=== test query: \"{query}\" ===")
    for model_name in models:
        print(f"\n--- {model_name} ---")
        try:
            results = ss.run_search(query, model_name)
        except Exception as e:
            print(f"  ERROR: {e}")
            continue
        if not results:
            print("  no results returned")
            continue
        for r in results[:3]:
            preview = r["content"][:100].replace("\n", " ")
            print(f"  docno={r['docno']}  score={r['score']:.4f}  title={r['title']!r}")
            print(f"    {preview}...")


if __name__ == "__main__":
    docstore, bert_emb, elmo_emb = check_counts()
    check_nans(bert_emb, elmo_emb)
    check_sample_content(docstore)

    models_to_test = list(ss.LEXICAL_MODELS) + ["BERT"]
    if elmo_emb:
        models_to_test.append("Elmo")

    check_search("earthquake", models_to_test)