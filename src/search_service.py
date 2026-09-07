import os
import pickle
import numpy as np
import pyterrier as pt

from src.text_processing import preprocess

INDEX_DIR = os.path.abspath("index/pyterrier_index")
DOCSTORE_PATH = "index/docstore.pkl"
BERT_EMB_PATH = "index/bert_embeddings.pkl"
ELMO_EMB_PATH = "index/elmo_embeddings.pkl"

LEXICAL_MODELS = ["BM25", "TF-IDF", "Hiemstra_LM", "PL2"]
EMBEDDING_MODELS = ["BERT", "Elmo"]
ALL_MODELS = LEXICAL_MODELS + EMBEDDING_MODELS

_index = None
_docstore = None
_bert_embeddings = None
_elmo_embeddings = None
_bert_model = None
_elmo_model = None


def load_everything():
    global _index, _docstore, _bert_embeddings, _elmo_embeddings

    if not pt.java.started():
        pt.init()

    _index = pt.IndexFactory.of(INDEX_DIR)

    with open(DOCSTORE_PATH, "rb") as f:
        _docstore = pickle.load(f)

    with open(BERT_EMB_PATH, "rb") as f:
        _bert_embeddings = pickle.load(f)

    try:
        with open(ELMO_EMB_PATH, "rb") as f:
            _elmo_embeddings = pickle.load(f)
    except FileNotFoundError:
        _elmo_embeddings = None


def _wmodel_name(model_name):
    return "TF_IDF" if model_name == "TF-IDF" else model_name


def get_lexical_model(model_name):
    controls = {"wmodel": _wmodel_name(model_name)}
    return pt.BatchRetrieve(_index, controls=controls, num_results=100)


def apply_relevance_feedback(model_name, processed_query):
    base_model = get_lexical_model(model_name)
    rm3 = pt.rewrite.RM3(_index, fb_terms=10, fb_docs=100)
    pipeline = base_model >> rm3
    expanded = pipeline.search(processed_query)
    if expanded.empty:
        return processed_query
    return expanded.iloc[0]["query"]


def cosine_similarity(v1, v2):
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)
    if denom == 0:
        return 0.0
    return float(np.dot(v1, v2) / denom)


def get_bert_model():
    global _bert_model
    if _bert_model is None:
        from sentence_transformers import SentenceTransformer
        _bert_model = SentenceTransformer('all-MiniLM-L6-v2')
    return _bert_model


def encode_bert(text):
    return get_bert_model().encode(text, convert_to_numpy=True)


def get_elmo_model():
    global _elmo_model
    if _elmo_model is None:
        import tensorflow_hub as hub
        _elmo_model = hub.load("https://tfhub.dev/google/elmo/3")
    return _elmo_model


def encode_elmo(text):
    import tensorflow as tf
    model = get_elmo_model()
    emb = model.signatures["default"](tf.constant([text]))["elmo"][0]
    return emb.numpy().mean(axis=0)


def search_embeddings(embeddings_dict, encode_fn, query_text, top_k=100):
    query_emb = encode_fn(query_text)
    scored = [
        (docno, cosine_similarity(doc_emb, query_emb))
        for docno, doc_emb in embeddings_dict.items()
    ]
    scored.sort(key=lambda pair: pair[1], reverse=True)
    return scored[:top_k]


def format_result(docno, score):
    doc = _docstore.get(docno, {})
    return {
        "docno": docno,
        "score": score,
        "title": doc.get("title", ""),
        "content": doc.get("text", "Document content not found"),
    }


def run_search(query_text, model_name):
    if _index is None:
        load_everything()

    if model_name not in ALL_MODELS:
        raise ValueError("Invalid model name: %s" % model_name)

    if model_name in LEXICAL_MODELS:
        processed_query = preprocess(query_text)
        expanded_query = apply_relevance_feedback(model_name, processed_query)
        model = get_lexical_model(model_name)
        results_df = model.search(expanded_query)
        return [
            format_result(row["docno"], row["score"])
            for _, row in results_df.iterrows()
        ]

    if model_name == "BERT":
        scored = search_embeddings(_bert_embeddings, encode_bert, query_text)
        return [format_result(docno, score) for docno, score in scored]

    if model_name == "Elmo":
        if _elmo_embeddings is None:
            raise ValueError(
                "Elmo embeddings were not built. Set BUILD_ELMO=True in "
                "build_index.py and rerun it, then try again."
            )
        scored = search_embeddings(_elmo_embeddings, encode_elmo, query_text)
        return [format_result(docno, score) for docno, score in scored]