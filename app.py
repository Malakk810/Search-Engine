from flask import Flask, render_template, request, jsonify

import src.search_service as search_service
import src.autocomplete as autocomplete

app = Flask(__name__)

MODEL_ORDER = ["BM25", "TF-IDF", "Hiemstra_LM", "PL2", "BERT", "Elmo"]


@app.route("/", methods=["GET", "POST"])
def search():
    results = []
    query = ""
    model = "BM25"
    error = None

    if request.method == "POST":
        query = request.form.get("query", "").strip()
        model = request.form.get("model", "BM25")

        if query:
            try:
                results = search_service.run_search(query, model)
            except Exception as e:
                error = str(e)

    if results:
        max_score = max(r["score"] for r in results)
        for r in results:
            if max_score > 0:
                pct = max(0.0, min(100.0, (r["score"] / max_score) * 100))
            else:
                pct = 0.0
            r["bar_pct"] = pct

    return render_template(
        "index.html",
        results=results,
        query=query,
        model=model,
        error=error,
        models=MODEL_ORDER,
    )


@app.route("/autocomplete", methods=["GET"])
def autocomplete_endpoint():
    prompt = request.args.get("q", "")
    if not prompt:
        return jsonify({"completion": ""})

    try:
        completion = autocomplete.suggest_completion(prompt, max_chars=10)
    except Exception:
        completion = ""

    return jsonify({"completion": completion})


if __name__ == "__main__":
    app.run(debug=True)