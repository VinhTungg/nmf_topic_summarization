"""
main.py - Điều phối pipeline + CLI (Người 4).

Pipeline: Văn bản -> Tiền xử lý -> TF-IDF -> NMF -> Topic -> Tóm tắt -> Đánh giá

Phiên bản CHÍNH THỨC: Sử dụng code thật từ Người 1-3, không còn mock data.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

# ---- Import module thật từ Người 1, 2, 3 ----
from src.preprocessing import load_documents, preprocess, split_sentences
from src.vectorizer import fit_tfidf, save_vectorizer, load_vectorizer
from src.topic_model import train_nmf, get_topics, get_document_topics
from src.summarizer import summarize

from src.evaluation import (
    build_doc_topic_table,
    build_k_table,
    build_summary_table,
    build_topic_table,
    evaluate_summary,
    evaluate_topics,
    topic_category_crosstab,
)
from src.utils import (
    load_model,
    plot_doc_topic_heatmap,
    plot_k_vs_error,
    save_model,
    save_table,
    save_text,
)


def _format_example(res, reference, metrics):
    lines = [
        "VÍ DỤ ĐẦY ĐỦ: Input -> Topic -> Summary",
        f"Input      : {res['original']}",
        f"Topic      : {res['topic_id']}  (từ khóa: {', '.join(res.get('keywords', []))})",
        f"Điểm câu   : {res.get('sentence_scores', [])}",
        f"Câu được chọn (chỉ số): {res.get('selected_indices', [])}",
        f"Summary    : {res['summary']}",
        f"Độ dài gốc / tóm tắt : {metrics['original_words']} / {metrics['summary_words']} từ",
        f"Compression ratio    : {metrics['compression_ratio']:.4f}",
    ]
    if reference and "rouge" in metrics:
        r = metrics["rouge"]
        lines.append(
            f"ROUGE-1/2/L (F1)     : {r['rouge1']['f1']:.4f} / {r['rouge2']['f1']:.4f} / {r['rougeL']['f1']:.4f}"
        )
    return "\n".join(lines)


def run_train(args):
    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(args.models_dir, exist_ok=True)

    # --- Bước 1: Đọc dữ liệu (Người 2) ------------------------------------
    print("[1/7] Đọc dữ liệu từ", args.data, "...")
    documents = load_documents(args.data)
    doc_ids = [d["doc_id"] for d in documents]
    categories = [d.get("category", "N/A") for d in documents]
    raw_texts = [d["text"] for d in documents]
    print(f"      -> Đọc được {len(documents)} văn bản.")

    # --- Bước 2: Tiền xử lý + TF-IDF (Người 2) ----------------------------
    print("[2/7] Tiền xử lý văn bản + Huấn luyện TF-IDF ...")
    processed_docs = [preprocess(t) for t in raw_texts]
    vectorizer, X = fit_tfidf(processed_docs)
    feature_names = list(vectorizer.get_feature_names_out())
    print(f"      -> Ma trận TF-IDF: {X.shape[0]} docs x {X.shape[1]} features.")

    # --- Bước 3: NMF (Người 1) + Thử nhiều k  ------------------------------
    print(f"[3/7] Huấn luyện NMF với k={args.n_topics} chủ đề ...")
    nmf_model = train_nmf(X, n_components=args.n_topics)
    W = get_document_topics(nmf_model, X)
    H = nmf_model.components_
    top_words = get_topics(nmf_model, feature_names, top_n=6)
    model_error = nmf_model.reconstruction_err_
    print(f"      -> Sai số tái tạo: {model_error:.4f}")

    print("[4/7] Thử nghiệm nhiều giá trị k ...")
    k_min, k_max = 2, min(args.n_topics + 3, len(documents))
    if k_max < k_min: k_max = k_min
    k_errors = {}
    for k in range(k_min, k_max + 1):
        m = train_nmf(X, n_components=k)
        k_errors[k] = m.reconstruction_err_
        print(f"      k={k}: error={m.reconstruction_err_:.4f}")

    # --- Bước 4: Tóm tắt từng văn bản (Người 3) ---------------------------
    print(f"[5/7] Tóm tắt {len(documents)} văn bản (mỗi văn bản lấy {args.num_sentences} câu) ...")
    results = []
    for d in documents:
        res = summarize(
            text=d["text"],
            vectorizer=vectorizer,
            nmf_model=nmf_model,
            num_sentences=args.num_sentences,
        )
        results.append(res)

    references = []
    for text in raw_texts:
        sents = split_sentences(text)
        if len(sents) >= 2:
            ref = sents[0] + " " + sents[-1]
        else:
            ref = text
        references.append(ref)

    # --- Bước 5: Đánh giá (Người 4) ----------------------------------------
    print("[6/7] Đánh giá chỉ số chất lượng ...")
    topic_metrics = evaluate_topics(X, W, H, model_error=model_error)
    summary_table = build_summary_table(results, references, doc_ids, categories)
    topic_table = build_topic_table(top_words)
    doc_topic_table = build_doc_topic_table(W, doc_ids, categories)
    k_table = build_k_table(k_errors)
    crosstab = topic_category_crosstab(doc_topic_table["dominant_topic"], categories)

    # --- Bước 6: Lưu kết quả -----------------------------------------------
    print("[7/7] Lưu kết quả ...")
    r = args.results_dir
    save_table(k_table, os.path.join(r, "k_reconstruction_error.csv"))
    save_table(topic_table, os.path.join(r, "topic_keywords.csv"))
    save_table(doc_topic_table, os.path.join(r, "document_topics.csv"))
    save_table(summary_table, os.path.join(r, "summary_metrics.csv"))
    save_table(crosstab.reset_index(), os.path.join(r, "topic_vs_category.csv"))
    plot_k_vs_error(k_table, os.path.join(r, "k_vs_error.png"))
    plot_doc_topic_heatmap(W, doc_ids, os.path.join(r, "doc_topic_heatmap.png"))

    example = _format_example(results[0], references[0], evaluate_summary(
        results[0]["original"], results[0]["summary"], references[0]))
    save_text(example, os.path.join(r, "example.txt"))

    save_vectorizer(vectorizer, os.path.join(args.models_dir, "tfidf.pkl"))
    save_model(nmf_model, os.path.join(args.models_dir, "nmf.pkl"))

    # --- In ra màn hình ---------------------------------------------------
    print("\n" + "=" * 70)
    print("KẾT QUẢ PIPELINE HOÀN CHỈNH (DỮ LIỆU THẬT)")
    print("=" * 70)
    print("\n[1] Chỉ số mô hình chủ đề")
    for key, val in topic_metrics.items():
        print(f"    {key}: {val}")
    print("\n[2] Bảng k - reconstruction error\n", k_table.to_string(index=False))
    print("\n[3] Top keywords theo topic\n", topic_table.to_string(index=False))
    print("\n[4] Topic chiếm ưu thế x nhãn tham chiếu\n", crosstab.to_string())
    print("\n[5] Chỉ số tóm tắt\n", summary_table.to_string(index=False))
    print("\n[6]", example)
    print(f"\nĐã lưu bảng/biểu đồ vào '{r}/' và mô hình vào '{args.models_dir}/'.")


def run_summarize(args):
    if not args.text:
        sys.exit("Thiếu --text khi dùng --mode summarize.")

    # Tải vectorizer và NMF model đã train
    vectorizer = load_vectorizer(os.path.join(args.models_dir, "tfidf.pkl"))
    nmf_model = load_model(os.path.join(args.models_dir, "nmf.pkl"))

    result = summarize(
        text=args.text,
        vectorizer=vectorizer,
        nmf_model=nmf_model,
        num_sentences=args.num_sentences,
    )

    metrics = evaluate_summary(result["original"], result["summary"])
    print(f"Topic phát hiện : {result['topic_id']}  (từ khóa: {', '.join(result.get('keywords', []))})")
    print(f"Điểm từng câu   : {result.get('sentence_scores', [])}")
    print(f"Câu được chọn   : {result.get('selected_indices', [])}")
    print(f"Tóm tắt         : {result['summary']}")
    print(f"Độ dài gốc/tóm tắt: {metrics['original_words']}/{metrics['summary_words']} từ, "
          f"compression ratio = {metrics['compression_ratio']:.4f}")


def main():
    # Windows console mặc định không phải UTF-8
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    p = argparse.ArgumentParser(description="NMF topic modeling + tóm tắt trích xuất tiếng Việt")
    p.add_argument("--mode", required=True, choices=["train", "summarize"])
    p.add_argument("--data", default="data/sample_documents.csv")
    p.add_argument("--n_topics", type=int, default=5)
    p.add_argument("--num_sentences", type=int, default=2)
    p.add_argument("--text", default=None)
    p.add_argument("--models_dir", default="models")
    p.add_argument("--results_dir", default="results")
    args = p.parse_args()

    if args.mode == "train":
        run_train(args)
    else:
        run_summarize(args)


if __name__ == "__main__":
    main()
