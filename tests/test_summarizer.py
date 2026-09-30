"""
Unit tests cho module tóm tắt văn bản trích xuất (src/summarizer.py).
Trách nhiệm: Người 3 (Topic-based Extractive Summarization).
"""

import sys
import unittest
from pathlib import Path

# Đảm bảo mã hóa UTF-8 trên Windows
if sys.platform == "win32":
    try:
        if sys.stdout and hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
        if sys.stderr and hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Thêm thư mục gốc vào sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF

from src.summarizer import (
    normalize_distribution,
    get_sentence_topics,
    score_sentences,
    remove_redundancy,
    summarize,
    sentence_details_to_dataframe,
    split_sentences,
    preprocess
)


class TestSummarizer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Tạo tập huấn luyện nhỏ gồm 2 chủ đề rõ ràng: Công nghệ và Thể thao
        cls.train_corpus = [
            "trí tuệ nhân tạo ai và học máy công nghệ phần mềm tự động hóa",
            "mô hình trí tuệ nhân tạo ai phân tích dữ liệu phát triển rất nhanh",
            "các doanh nghiệp ứng dụng ai công nghệ vào sản xuất tự động hóa",
            "bóng đá thể thao cầu thủ sân cỏ trận đấu bàn thắng giải đấu",
            "giải đấu bóng đá vô địch câu lạc bộ cầu thủ huấn luyện viên",
            "trận đấu thể thao kịch tính trên sân cỏ bàn thắng đẹp mắt"
        ]
        cls.vectorizer = TfidfVectorizer()
        cls.X_tfidf = cls.vectorizer.fit_transform(cls.train_corpus)
        cls.nmf_model = NMF(n_components=2, random_state=42, init="nndsvda", max_iter=400)
        cls.nmf_model.fit(cls.X_tfidf)

    def test_score_dominant(self):
        """Kiểm tra chấm điểm theo chủ đề chiếm ưu thế (Method 1)."""
        doc_topic = np.array([0.8, 0.2])  # dominant_topic = 0
        sent_topics = np.array([
            [0.9, 0.1],
            [0.3, 0.7],
            [0.5, 0.5]
        ])
        scores = score_sentences(doc_topic, sent_topics, method="dominant", normalize_topics=True)
        # dominant_topic = 0, điểm câu phải bằng activation tại topic 0 sau khi chuẩn hóa
        self.assertEqual(len(scores), 3)
        self.assertAlmostEqual(scores[0], 0.9, places=4)
        self.assertAlmostEqual(scores[1], 0.3, places=4)
        self.assertAlmostEqual(scores[2], 0.5, places=4)
        self.assertGreater(scores[0], scores[1])

    def test_score_weighted(self):
        """Kiểm tra chấm điểm theo trọng số toàn bộ chủ đề (Method 2 - weighted dot product)."""
        doc_topic = np.array([0.7, 0.3])
        sent_topics = np.array([
            [0.5, 0.5],
            [1.0, 0.0]
        ])
        scores = score_sentences(doc_topic, sent_topics, method="weighted", normalize_topics=True)
        # Câu 0: 0.7*0.5 + 0.3*0.5 = 0.50
        # Câu 1: 0.7*1.0 + 0.3*0.0 = 0.70
        self.assertEqual(len(scores), 2)
        self.assertAlmostEqual(scores[0], 0.50, places=4)
        self.assertAlmostEqual(scores[1], 0.70, places=4)

    def test_redundancy(self):
        """Kiểm tra loại bỏ câu trùng lặp ngữ nghĩa bằng Cosine Similarity."""
        # Giả lập ma trận TF-IDF 3 câu: câu 0 và câu 1 giống hệt nhau
        vec_matrix = np.array([
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0]
        ])
        ranked_indices = [0, 1, 2]
        # Ngưỡng 0.7 -> câu 1 có độ tương đồng 1.0 với câu 0, phải bị loại
        selected = remove_redundancy(
            ranked_indices,
            vec_matrix,
            num_sentences=2,
            similarity_threshold=0.7,
            allow_fill=True
        )
        self.assertEqual(selected, [0, 2])

    def test_single_sentence(self):
        """Kiểm tra văn bản chỉ có đúng 1 câu."""
        text = "Trí tuệ nhân tạo đang phát triển rất mạnh mẽ."
        result = summarize(text, self.vectorizer, self.nmf_model, num_sentences=3)
        self.assertEqual(result["num_original_sentences"], 1)
        self.assertEqual(len(result["selected_indices"]), 1)
        self.assertEqual(result["summary"], text)

    def test_invalid_threshold(self):
        """Kiểm tra báo lỗi khi threshold nằm ngoài khoảng [0, 1]."""
        with self.assertRaises(ValueError):
            remove_redundancy([0, 1], np.eye(2), similarity_threshold=1.5)
        with self.assertRaises(ValueError):
            remove_redundancy([0, 1], np.eye(2), similarity_threshold=-0.1)

    def test_empty_text(self):
        """Kiểm tra báo lỗi khi văn bản đầu vào rỗng."""
        with self.assertRaises(ValueError):
            summarize("", self.vectorizer, self.nmf_model)
        with self.assertRaises(ValueError):
            summarize("    \n\t  ", self.vectorizer, self.nmf_model)

    def test_num_sentences_greater_than_total(self):
        """Kiểm tra khi num_sentences yêu cầu lớn hơn tổng số câu văn bản."""
        text = (
            "Trí tuệ nhân tạo đang thay đổi thế giới. "
            "Nhiều doanh nghiệp ứng dụng AI thành công. "
            "Công nghệ này mang lại nhiều lợi ích."
        )
        result = summarize(text, self.vectorizer, self.nmf_model, num_sentences=10)
        self.assertEqual(result["num_original_sentences"], 3)
        self.assertEqual(len(result["selected_indices"]), 3)

    def test_repeated_sentences(self):
        """Test 6: Văn bản có câu lặp lại, filter phải không chọn câu trùng lặp."""
        text = (
            "AI đang phát triển rất nhanh. "
            "AI đang phát triển rất nhanh. "
            "Các doanh nghiệp ứng dụng AI vào sản xuất."
        )
        result = summarize(
            text,
            self.vectorizer,
            self.nmf_model,
            num_sentences=2,
            similarity_threshold=0.7
        )
        # Câu 0 và câu 1 trùng lặp hoàn toàn, do đó hai câu được chọn phải là câu 0 và câu 2 (hoặc 1 và 2)
        selected = result["selected_indices"]
        self.assertEqual(len(selected), 2)
        self.assertFalse(0 in selected and 1 in selected)

    def test_unknown_vocabulary(self):
        """Test 7: Văn bản chứa từ hoàn toàn ngoài từ vựng TF-IDF đã học."""
        text = "Xyzqwerty abcdefgh mkljhyt."
        result = summarize(text, self.vectorizer, self.nmf_model, num_sentences=1)
        self.assertIn("warning", result)
        self.assertEqual(result["detected_topic"], 0)
        self.assertEqual(len(result["selected_indices"]), 1)
        self.assertEqual(result["summary"], text)

    def test_invalid_method(self):
        """Kiểm tra báo lỗi khi truyền method không hợp lệ."""
        text = "Trí tuệ nhân tạo đang phát triển nhanh. Doanh nghiệp cần đổi mới sáng tạo."
        with self.assertRaises(ValueError):
            summarize(text, self.vectorizer, self.nmf_model, scoring_method="invalid_method")

    def test_none_models(self):
        """Kiểm tra báo lỗi khi vectorizer hoặc nmf_model là None."""
        text = "Trí tuệ nhân tạo đang phát triển."
        with self.assertRaises(ValueError):
            summarize(text, None, self.nmf_model)
        with self.assertRaises(ValueError):
            summarize(text, self.vectorizer, None)

    def test_original_order_preserved(self):
        """Kiểm tra thứ tự câu gốc luôn được giữ nguyên trong bản tóm tắt."""
        text = (
            "Câu số một giới thiệu vấn đề chung. "
            "Câu số hai nói về công nghệ trí tuệ nhân tạo và học máy rất nổi bật. "
            "Câu số ba là kết luận tóm lại toàn bộ nội dung."
        )
        result = summarize(text, self.vectorizer, self.nmf_model, num_sentences=2)
        # Các chỉ số câu được chọn phải được sắp xếp tăng dần
        self.assertEqual(result["selected_indices"], sorted(result["selected_indices"]))

    def test_sentence_details_structure(self):
        """Kiểm tra cấu trúc chi tiết sentence_details phục vụ báo cáo và đánh giá."""
        text = "Công nghệ phần mềm phát triển. Dữ liệu lớn rất quan trọng."
        result = summarize(text, self.vectorizer, self.nmf_model, num_sentences=1)
        self.assertIn("sentence_details", result)
        for detail in result["sentence_details"]:
            self.assertIn("index", detail)
            self.assertIn("sentence", detail)
            self.assertIn("dominant_topic", detail)
            self.assertIn("topic_distribution", detail)
            self.assertIn("score", detail)
            self.assertIn("selected", detail)

    def test_dataframe_conversion(self):
        """Kiểm tra hàm chuyển đổi kết quả sang Pandas DataFrame."""
        text = "Trí tuệ nhân tạo phát triển. Doanh nghiệp ứng dụng công nghệ."
        result = summarize(text, self.vectorizer, self.nmf_model, num_sentences=1)
        df = sentence_details_to_dataframe(result)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 2)
        expected_cols = {"index", "sentence", "dominant_topic", "score", "selected"}
        self.assertTrue(expected_cols.issubset(df.columns))

    def test_evaluation_contract_compatibility(self):
        """Kiểm tra tính tương thích các trường đầu ra cho Người 4 (evaluation.py & main.py)."""
        text = "Trí tuệ nhân tạo phát triển. Doanh nghiệp ứng dụng công nghệ."
        result = summarize(text, self.vectorizer, self.nmf_model, num_sentences=1)
        expected_contract_keys = ["original", "summary", "topic_id", "keywords", "sentence_scores", "selected_indices"]
        for key in expected_contract_keys:
            self.assertIn(key, result, f"Thiếu trường '{key}' theo quy ước chung của Người 4")
        self.assertEqual(result["original"], text)
        self.assertEqual(result["topic_id"], result["detected_topic"])
        self.assertEqual(result["keywords"], result["topic_keywords"])
        self.assertEqual(len(result["sentence_scores"]), len(result["sentences"]))
        self.assertEqual(result["sentence_scores"][0], result["sentence_details"][0]["score"])


if __name__ == "__main__":
    unittest.main()

