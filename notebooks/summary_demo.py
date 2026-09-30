"""
Kịch bản minh họa (Demo Script) cho Người 3:
Topic-based Extractive Summarization bằng NMF trên dữ liệu tiếng Việt.

Chức năng:
1. Đọc dữ liệu mẫu từ data/sample_documents.csv (không dùng nhãn category).
2. Huấn luyện demo TfidfVectorizer và NMF (k=5 chủ đề) bằng API của Người 1.
3. Thực hiện tóm tắt văn bản thử nghiệm mới (chưa có trong tập huấn luyện).
4. So sánh kết quả giữa 2 phương pháp tính điểm:
   - Method 1: Dominant Topic Score
   - Method 2: Weighted Topic Similarity
5. Xuất bảng điểm chi tiết ra file results/sentence_scores.csv (utf-8-sig).
6. Lưu kết quả tóm tắt chi tiết vào file results/summary_demo.txt.
"""

import sys
import os
from pathlib import Path

# Đảm bảo terminal Windows hiển thị tiếng Việt UTF-8 không lỗi font/charmap
if sys.platform == "win32":
    try:
        if sys.stdout and hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
        if sys.stderr and hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Thêm thư mục gốc của project vào sys.path để hỗ trợ chạy từ bất kỳ thư mục nào
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from src.topic_model import train_nmf, get_topics, get_top_words
from src.summarizer import (
    preprocess,
    summarize,
    sentence_details_to_dataframe
)


def run_demo():
    print("=" * 80)
    print(" DEMO TÓM TẮT VĂN BẢN DỰA TRÊN CHỦ ĐỀ NMF (NGƯỜI 3 - TIẾNG VIỆT) ")
    print("=" * 80)

    # 1. Tải dữ liệu huấn luyện mẫu
    data_path = project_root / "data" / "sample_documents.csv"
    if not data_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file dữ liệu mẫu tại: {data_path}")

    print(f"\n[1/5] Đang đọc dữ liệu huấn luyện từ {data_path}...")
    df = pd.read_csv(data_path)
    print(f"      -> Đã tải {len(df)} văn bản mẫu.")

    # 2. Huấn luyện mô hình TF-IDF và NMF demo (k=5)
    # Tuyệt đối không dùng nhãn category cho mô hình
    print("\n[2/5] Tiền xử lý và huấn luyện TF-IDF & NMF (k=5 chủ đề)...")
    corpus = [preprocess(text) for text in df["text"]]
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus)

    # Sử dụng hàm train_nmf từ module Người 1
    k_topics = 5
    nmf_model = train_nmf(
        tfidf_matrix,
        n_components=k_topics,
        random_state=42,
        init="nndsvda",
        max_iter=400
    )
    feature_names = vectorizer.get_feature_names_out()

    print("      -> Huấn luyện NMF thành công! Danh sách các chủ đề đã học:")
    topics_dict = get_topics(nmf_model, feature_names, top_n=5)
    for tid, words in topics_dict.items():
        print(f"         * Chủ đề {tid}: {', '.join(words)}")

    # 3. Văn bản kiểm thử mới (Section 41)
    test_document = (
        "Trí tuệ nhân tạo ngày càng được ứng dụng rộng rãi trong doanh nghiệp. "
        "Nhiều công ty sử dụng các mô hình học máy để phân tích dữ liệu khách hàng. "
        "Các hệ thống AI cũng giúp tự động hóa những công việc lặp lại. "
        "Tuy nhiên doanh nghiệp cần quan tâm đến bảo mật dữ liệu khi triển khai công nghệ. "
        "Việc đào tạo nhân viên sử dụng công cụ số cũng đóng vai trò quan trọng."
    )

    print("\n[3/5] Thực hiện tóm tắt văn bản kiểm thử mới...")

    # Chạy phương pháp Weighted Topic Similarity (Mặc định)
    result_weighted = summarize(
        text=test_document,
        vectorizer=vectorizer,
        nmf_model=nmf_model,
        num_sentences=2,
        scoring_method="weighted",
        similarity_threshold=0.7,
        normalize_topics=True,
        feature_names=feature_names,
        top_n_keywords=5
    )

    # Chạy phương pháp Dominant Topic Score để so sánh
    result_dominant = summarize(
        text=test_document,
        vectorizer=vectorizer,
        nmf_model=nmf_model,
        num_sentences=2,
        scoring_method="dominant",
        similarity_threshold=0.7,
        normalize_topics=True,
        feature_names=feature_names,
        top_n_keywords=5
    )

    # 4. Hiển thị kết quả ra Terminal theo định dạng yêu cầu
    print("\n" + "=" * 80)
    print("===== DOCUMENT =====")
    print(test_document)

    print("\n===== DETECTED TOPIC =====")
    print(f"Topic {result_weighted['detected_topic']}")
    print(f"Phân phối chủ đề tài liệu: {result_weighted['document_topic_distribution']}")

    print("\n===== TOPIC KEYWORDS =====")
    print(", ".join(result_weighted["topic_keywords"]))

    print("\n===== SENTENCE SCORES (WEIGHTED METHOD) =====")
    for d in result_weighted["sentence_details"]:
        print(f"[{d['index']}] Score: {d['score']:.4f}")
        print(f"Sentence: {d['sentence']}\n")

    print("===== SO SÁNH HAI CÁCH SENTENCE SCORING =====")
    print(f"{'Idx':<4} | {'Dominant Score':<15} | {'Weighted Score':<15} | {'Nội dung câu'}")
    print("-" * 80)
    for i in range(len(result_weighted["sentence_details"])):
        d_dom = result_dominant["sentence_details"][i]
        d_wei = result_weighted["sentence_details"][i]
        print(f"{i:<4} | {d_dom['score']:<15.4f} | {d_wei['score']:<15.4f} | {d_wei['sentence'][:45]}...")

    print("\n===== SELECTED SENTENCES =====")
    print(f"Weighted method chọn các câu: {result_weighted['selected_indices']}")
    print(f"Dominant method chọn các câu: {result_dominant['selected_indices']}")

    print("\n===== SUMMARY (WEIGHTED) =====")
    print(result_weighted["summary"])

    print("\n===== SUMMARY (DOMINANT) =====")
    print(result_dominant["summary"])
    print("=" * 80)

    # 5. Lưu kết quả ra thư mục results/
    results_dir = project_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Xuất file results/sentence_scores.csv (utf-8-sig cho Excel tiếng Việt)
    df_scores = sentence_details_to_dataframe(result_weighted)
    # Bổ sung thêm cột điểm dominant để tiện đối chiếu trong báo cáo
    df_scores["dominant_score"] = [d["score"] for d in result_dominant["sentence_details"]]
    df_scores["selected_dominant"] = [d["selected"] for d in result_dominant["sentence_details"]]
    df_scores.rename(columns={"score": "weighted_score", "selected": "selected_weighted"}, inplace=True)
    
    # Thứ tự cột chuẩn và thân thiện
    final_cols = ["index", "sentence", "dominant_topic", "weighted_score", "selected_weighted", "dominant_score", "selected_dominant"]
    df_scores = df_scores[final_cols]
    
    csv_file = results_dir / "sentence_scores.csv"
    df_scores.to_csv(csv_file, index=False, encoding="utf-8-sig")
    print(f"\n[4/5] Đã lưu bảng điểm câu chi tiết vào: {csv_file}")

    # Xuất file results/summary_demo.txt (Section 29)
    txt_file = results_dir / "summary_demo.txt"
    with open(txt_file, "w", encoding="utf-8") as f:
        f.write("Original Document:\n")
        f.write(result_weighted["original_text"] + "\n\n")
        f.write(f"Detected Topic:\nTopic {result_weighted['detected_topic']}\n\n")
        f.write(f"Keywords:\n{', '.join(result_weighted['topic_keywords'])}\n\n")
        f.write(f"Selected Sentence Indices:\n{result_weighted['selected_indices']}\n\n")
        f.write("Summary:\n")
        f.write(result_weighted["summary"] + "\n")

    print(f"[5/5] Đã lưu bản tóm tắt demo vào: {txt_file}")
    print("\n>>> DEMO HOÀN TẤT THÀNH CÔNG! <<<")


if __name__ == "__main__":
    run_demo()
