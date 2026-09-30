"""
Module tóm tắt văn bản trích xuất dựa trên chủ đề (Topic-based Extractive Summarization).
Trách nhiệm: Người 3 (Topic-based Extractive Summarization).

Cơ sở lý thuyết & Quy trình thuật toán:
    1. Tách câu tiếng Việt (split_sentences) và giữ nguyên câu gốc cùng dấu câu.
    2. Tiền xử lý (preprocess) phục vụ vector hóa.
    3. Biến đổi TF-IDF bằng vectorizer đã huấn luyện (vectorizer.transform). TUYỆT ĐỐI KHÔNG FIT LẠI.
    4. Chiếu vector document và các câu sang không gian chủ đề NMF (nmf_model.transform). TUYỆT ĐỐI KHÔNG TRAIN LẠI.
    5. Xác định chủ đề chiếm ưu thế (dominant_topic) và phân phối chủ đề của document.
    6. Chấm điểm từng câu dựa trên phân phối chủ đề:
       - Method 1 ('dominant'): Score(S_i) = SentenceTopic[i, dominant_topic]
       - Method 2 ('weighted'): Score(S_i) = Σ_k DocTopic[k] × SentenceTopic[i, k]
    7. Xếp hạng câu giảm dần theo điểm số.
    8. Lọc câu trùng lặp nội dung (remove_redundancy) bằng Cosine Similarity trên ma trận TF-IDF.
    9. Chọn Top-N câu quan trọng nhất.
    10. Phục hồi thứ tự xuất hiện ban đầu của các câu được chọn (preserve original order).
    11. Ghép thành bản tóm tắt trích xuất (Extractive Summary).
"""

from typing import List, Dict, Tuple, Optional, Any, Union
import sys
import re
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Tự động hỗ trợ hiển thị tiếng Việt trên terminal Windows
if sys.platform == "win32":
    try:
        if sys.stdout and hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
        if sys.stderr and hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Import các hàm từ Người 1 (NMF / Topic Modeling)
try:
    from src.topic_model import get_top_words, predict_topic
except ImportError:
    try:
        from topic_model import get_top_words, predict_topic
    except ImportError:
        get_top_words = None
        predict_topic = None

# Import API dự kiến từ Người 2 (Preprocessing)
try:
    from src.preprocessing import split_sentences as _p2_split_sentences
    from src.preprocessing import preprocess as _p2_preprocess
except ImportError:
    try:
        from preprocessing import split_sentences as _p2_split_sentences
        from preprocessing import preprocess as _p2_preprocess
    except ImportError:
        _p2_split_sentences = None
        _p2_preprocess = None


# =====================================================================
# FALLBACK PREPROCESSING (CHỈ DÙNG CHO DEVELOPMENT VÀ TEST ĐỘC LẬP)
# Khi Người 2 hoàn thiện module src/preprocessing.py, hệ thống sẽ tự
# động ưu tiên dùng split_sentences() và preprocess() từ Người 2.
# =====================================================================

def _fallback_split_sentences(text: str) -> List[str]:
    """
    Tách câu tiếng Việt dự phòng khi module của Người 2 chưa sẵn sàng.
    Ưu tiên sử dụng underthesea.sent_tokenize, có kèm fallback regex.
    """
    if not text or not text.strip():
        return []
    try:
        import underthesea
        raw_sentences = underthesea.sent_tokenize(text)
    except Exception:
        # Regex fallback nếu underthesea gặp sự cố
        raw_sentences = re.split(r"(?<=[.!?])\s+|\n+", text)

    sentences = [s.strip() for s in raw_sentences if s and s.strip()]
    return sentences


def _fallback_preprocess(text: str) -> str:
    """
    Tiền xử lý văn bản dự phòng khi module của Người 2 chưa sẵn sàng.
    Sử dụng underthesea word_tokenize gán nhãn từ ghép tiếng Việt (dấu gạch dưới).
    """
    if not text or not text.strip():
        return ""
    try:
        import underthesea
        return underthesea.word_tokenize(text.lower().strip(), format="text")
    except Exception:
        return text.lower().strip()


def split_sentences(text: str) -> List[str]:
    """
    Tách văn bản thành danh sách các câu tiếng Việt riêng biệt.
    Ưu tiên gọi split_sentences từ src.preprocessing nếu khả dụng.

    Tham số:
        text (str): Văn bản tiếng Việt cần tách câu.

    Trả về:
        List[str]: Danh sách các câu nguyên bản (giữ nguyên dấu câu và cấu trúc gốc).
    """
    if _p2_split_sentences is not None and callable(_p2_split_sentences):
        try:
            sents = _p2_split_sentences(text)
            if sents and len(sents) > 0:
                cleaned = [s.strip() for s in sents if s and s.strip()]
                if cleaned:
                    return cleaned
        except Exception:
            pass
    return _fallback_split_sentences(text)


def preprocess(text: str) -> str:
    """
    Tiền xử lý văn bản/câu phục vụ vector hóa TF-IDF.
    Ưu tiên gọi preprocess từ src.preprocessing nếu khả dụng.

    Tham số:
        text (str): Chuỗi văn bản cần tiền xử lý.

    Trả về:
        str: Chuỗi văn bản đã được chuẩn hóa/tách từ.
    """
    if _p2_preprocess is not None and callable(_p2_preprocess):
        try:
            res = _p2_preprocess(text)
            if res is not None:
                return str(res)
        except Exception:
            pass
    return _fallback_preprocess(text)


# =====================================================================
# CÁC HÀM CỐT LÕI CỦA SUMMARIZER (NGƯỜI 3)
# =====================================================================

def normalize_distribution(data: np.ndarray) -> np.ndarray:
    """
    Chuẩn hóa L1 phân phối chủ đề để tổng các xác suất/trọng số bằng 1.0.
    Nếu vector toàn số 0, bảo toàn vector số 0 để tránh chia cho 0.
    Chỉ dùng chuẩn hóa để tính điểm câu (scoring), không làm thay đổi mô hình NMF.

    Tham số:
        data (np.ndarray): Vector 1D hoặc ma trận 2D cần chuẩn hóa.

    Trả về:
        np.ndarray: Dữ liệu đã chuẩn hóa L1.
    """
    arr = np.asarray(data, dtype=float)
    if arr.ndim == 1:
        total = float(np.sum(arr))
        if total > 0.0:
            return arr / total
        return arr.copy()
    elif arr.ndim == 2:
        row_sums = np.sum(arr, axis=1, keepdims=True)
        normalized = np.zeros_like(arr, dtype=float)
        mask = (row_sums.ravel() > 0.0)
        if np.any(mask):
            normalized[mask] = arr[mask] / row_sums[mask]
        return normalized
    else:
        raise ValueError(f"normalize_distribution chỉ hỗ trợ mảng 1D hoặc 2D, nhận được ndim={arr.ndim}")


def get_sentence_topics(sentence_tfidf_matrix: Any, nmf_model: Any) -> np.ndarray:
    """
    Chiếu ma trận đặc trưng TF-IDF của các câu sang không gian chủ đề NMF.
    TUYỆT ĐỐI KHÔNG TRAIN LẠI NMF - chỉ dùng nmf_model.transform.

    Tham số:
        sentence_tfidf_matrix: Ma trận TF-IDF của các câu (scipy.sparse hoặc np.ndarray).
        nmf_model: Mô hình NMF đã huấn luyện từ trước (sklearn.decomposition.NMF).

    Trả về:
        np.ndarray: Ma trận kích thước (num_sentences x num_topics) biểu diễn mức kích hoạt chủ đề của từng câu.
    """
    if sentence_tfidf_matrix is None or nmf_model is None:
        raise ValueError("sentence_tfidf_matrix và nmf_model không được để trống (None).")

    if hasattr(sentence_tfidf_matrix, "shape") and sentence_tfidf_matrix.shape[0] == 0:
        n_components = getattr(nmf_model, "n_components", 0)
        return np.empty((0, n_components), dtype=float)

    sentence_topic_matrix = nmf_model.transform(sentence_tfidf_matrix)
    return np.asarray(sentence_topic_matrix, dtype=float)


def score_sentences(
    document_topic_distribution: np.ndarray,
    sentence_topic_matrix: np.ndarray,
    method: str = "weighted",
    normalize_topics: bool = True
) -> np.ndarray:
    """
    Chấm điểm độ quan trọng của từng câu văn dựa trên phân phối chủ đề.

    Hỗ trợ 2 phương pháp:
        Method 1 ('dominant'):
            Score(S_i) = SentenceTopic[i, dominant_topic]
            Trong đó dominant_topic = argmax(document_topic_distribution).

        Method 2 ('weighted' - mặc định):
            Score(S_i) = Σ_k DocTopic[k] × SentenceTopic[i, k]
            Tích vô hướng giữa phân phối chủ đề tài liệu và phân phối chủ đề câu.

    Tham số:
        document_topic_distribution (np.ndarray): Mảng 1D phân phối chủ đề của document (kích thước k).
        sentence_topic_matrix (np.ndarray): Mảng 2D kích thước (num_sentences x k) từ get_sentence_topics.
        method (str): 'weighted' (mặc định) hoặc 'dominant'.
        normalize_topics (bool): Bật/tắt chuẩn hóa L1 phân phối trước khi chấm điểm.

    Trả về:
        np.ndarray: Mảng 1D chứa điểm số (score) của từng câu.
    """
    if method not in ("dominant", "weighted"):
        raise ValueError(
            f"Phương pháp chấm điểm không hợp lệ: '{method}'. "
            "Chỉ chấp nhận 'dominant' hoặc 'weighted'."
        )

    doc_dist = np.asarray(document_topic_distribution, dtype=float)
    if doc_dist.ndim != 1:
        doc_dist = doc_dist.ravel()

    sent_matrix = np.asarray(sentence_topic_matrix, dtype=float)
    if sent_matrix.ndim == 1:
        if sent_matrix.size == 0:
            return np.array([], dtype=float)
        sent_matrix = sent_matrix.reshape(1, -1)
    elif sent_matrix.ndim == 2 and sent_matrix.shape[0] == 0:
        return np.array([], dtype=float)
    elif sent_matrix.ndim != 2:
        raise ValueError(f"sentence_topic_matrix phải là mảng 2D, nhận được ndim={sent_matrix.ndim}")

    num_topics_doc = doc_dist.shape[0]
    num_topics_sent = sent_matrix.shape[1]

    if num_topics_doc != num_topics_sent:
        raise ValueError(
            f"Số lượng chủ đề không khớp: tài liệu có {num_topics_doc} chủ đề, "
            f"nhưng các câu có {num_topics_sent} chủ đề."
        )

    # Chuẩn hóa nếu bật cờ normalize_topics
    if normalize_topics:
        norm_doc = normalize_distribution(doc_dist)
        norm_sent = normalize_distribution(sent_matrix)
    else:
        norm_doc = doc_dist
        norm_sent = sent_matrix

    if method == "dominant":
        dominant_topic = int(np.argmax(doc_dist)) if np.sum(doc_dist) > 0 else 0
        scores = norm_sent[:, dominant_topic]
    else:  # method == "weighted"
        scores = norm_sent @ norm_doc

    return np.asarray(scores, dtype=float)


def remove_redundancy(
    ranked_indices: List[int],
    sentence_tfidf_matrix: Any,
    num_sentences: int = 3,
    similarity_threshold: float = 0.7,
    allow_fill: bool = True,
    sentence_texts: Optional[List[str]] = None
) -> List[int]:
    """
    Lọc bỏ các câu trùng lặp ngữ nghĩa dựa trên Cosine Similarity giữa các vector TF-IDF của câu.
    Không dùng topic distribution để lọc trùng lặp vì hai câu cùng chủ đề vẫn có thể chứa thông tin khác nhau.

    Thuật toán:
        1. Duyệt qua từng câu theo thứ tự xếp hạng (ranked_indices).
        2. Nếu chưa chọn câu nào -> chọn câu đầu tiên.
        3. Với câu ứng viên tiếp theo, tính Cosine Similarity với các câu đã chọn.
           (Nếu câu trùng lặp văn bản tuyệt đối với câu đã chọn thì similarity = 1.0).
        4. Nếu max_similarity < similarity_threshold -> chọn câu đó.
        5. Nếu max_similarity >= similarity_threshold -> bỏ qua câu vì trùng lặp.
        6. Nếu kết thúc vòng duyệt mà chưa đủ num_sentences (do lọc quá khắt khe) và allow_fill=True,
           bổ sung các câu bị loại theo thứ tự điểm cao nhất còn lại để đảm bảo đủ số câu.

    Tham số:
        ranked_indices (List[int]): Danh sách chỉ số câu đã xếp hạng theo score giảm dần.
        sentence_tfidf_matrix: Ma trận TF-IDF của các câu.
        num_sentences (int): Số lượng câu mục tiêu cần chọn.
        similarity_threshold (float): Ngưỡng độ tương đồng [0.0, 1.0]. Mặc định 0.7.
        allow_fill (bool): Cho phép bù câu nếu việc lọc khiến số câu thiếu so với mục tiêu.
        sentence_texts (Optional[List[str]]): Danh sách chuỗi văn bản của các câu (để hỗ trợ phát hiện câu trùng lặp 100%).

    Trả về:
        List[int]: Danh sách các chỉ số câu được chọn theo thứ tự RANKING (ưu tiên câu điểm cao).
    """
    if similarity_threshold < 0.0 or similarity_threshold > 1.0:
        raise ValueError(
            f"similarity_threshold phải nằm trong khoảng [0.0, 1.0], nhận được {similarity_threshold}"
        )
    if num_sentences <= 0:
        raise ValueError(
            f"num_sentences phải lớn hơn 0, nhận được {num_sentences}"
        )

    if not ranked_indices:
        return []

    target_count = min(num_sentences, len(ranked_indices))
    if len(ranked_indices) == 1:
        return [ranked_indices[0]]

    selected: List[int] = []
    skipped: List[int] = []

    for idx in ranked_indices:
        if len(selected) == 0:
            selected.append(idx)
            if len(selected) == target_count:
                break
            continue

        # Kiểm tra trùng lặp văn bản trực tiếp nếu có sentence_texts
        is_exact_duplicate = False
        if sentence_texts is not None and idx < len(sentence_texts):
            curr_str = sentence_texts[idx].strip().lower()
            for sel_idx in selected:
                if sel_idx < len(sentence_texts) and curr_str == sentence_texts[sel_idx].strip().lower():
                    is_exact_duplicate = True
                    break

        if is_exact_duplicate:
            max_sim = 1.0
        else:
            # Lấy vector TF-IDF của câu hiện tại và ma trận các câu đã chọn
            curr_vec = sentence_tfidf_matrix[idx : idx + 1]
            selected_matrix = sentence_tfidf_matrix[selected]

            sims = cosine_similarity(curr_vec, selected_matrix)
            sims_clean = np.nan_to_num(sims, nan=0.0)
            max_sim = float(np.max(sims_clean))

        if max_sim < similarity_threshold:
            selected.append(idx)
        else:
            skipped.append(idx)

        if len(selected) == target_count:
            break

    # Dự phòng: nếu lọc quá mạnh không đủ num_sentences, bổ sung các câu có điểm cao còn lại
    if allow_fill and len(selected) < target_count:
        for idx in skipped:
            selected.append(idx)
            if len(selected) == target_count:
                break

    return selected


def sentence_details_to_dataframe(result: Dict[str, Any]) -> pd.DataFrame:
    """
    Chuyển đổi danh sách sentence_details từ kết quả tóm tắt thành Pandas DataFrame.
    Hỗ trợ xuất báo cáo, phân tích và lưu CSV (utf-8-sig).

    Tham số:
        result (Dict[str, Any]): Kết quả trả về từ hàm summarize().

    Trả về:
        pd.DataFrame: Bảng chứa các cột [index, sentence, dominant_topic, score, selected].
    """
    details = result.get("sentence_details", [])
    records = []
    for d in details:
        records.append({
            "index": d["index"],
            "sentence": d["sentence"],
            "dominant_topic": d.get("dominant_topic", 0),
            "score": d["score"],
            "selected": d["selected"]
        })
    return pd.DataFrame(records)


def summarize(
    text: str,
    vectorizer: Any,
    nmf_model: Any,
    num_sentences: int = 3,
    scoring_method: str = "weighted",
    similarity_threshold: float = 0.7,
    normalize_topics: bool = True,
    feature_names: Optional[List[str]] = None,
    top_n_keywords: int = 5,
    custom_split_sentences: Optional[Any] = None,
    custom_preprocess: Optional[Any] = None
) -> Dict[str, Any]:
    """
    Hàm API chính thực hiện tóm tắt văn bản trích xuất dựa trên chủ đề (NMF Topic Modeling).

    Tuyệt đối tuân thủ nguyên tắc:
        - KHÔNG fit lại vectorizer.
        - KHÔNG train lại nmf_model.
        - GIỮ NGUYÊN câu gốc tiếng Việt cho output (không làm mất dấu câu).
        - GIỮ THỨ TỰ CÂU XUẤT HIỆN BAN ĐẦU trong bản tóm tắt cuối cùng.

    Tham số:
        text (str): Văn bản tiếng Việt cần tóm tắt.
        vectorizer: Mô hình TfidfVectorizer đã được fit từ trước bởi Người 2.
        nmf_model: Mô hình NMF đã được fit từ trước bởi Người 1.
        num_sentences (int): Số lượng câu cần trích xuất cho bản tóm tắt (mặc định 3).
        scoring_method (str): Phương pháp tính điểm câu ('weighted' hoặc 'dominant'). Mặc định 'weighted'.
        similarity_threshold (float): Ngưỡng tương đồng cosine để loại câu trùng lặp (mặc định 0.7).
        normalize_topics (bool): Có chuẩn hóa L1 phân phối chủ đề trước khi chấm điểm hay không (mặc định True).
        feature_names (Optional[List[str]]): Danh sách tên từ vựng (nếu None sẽ tự lấy từ vectorizer).
        top_n_keywords (int): Số lượng từ khóa đặc trưng cần trích xuất của topic (mặc định 5).
        custom_split_sentences (Optional[Callable]): Hàm tách câu tùy biến (nếu truyền vào).
        custom_preprocess (Optional[Callable]): Hàm tiền xử lý tùy biến (nếu truyền vào).

    Trả về:
        Dict[str, Any]: Từ điển chứa đầy đủ thông tin tóm tắt và metadata chi tiết phục vụ đánh giá & báo cáo:
            - original_text (str)
            - sentences (List[str])
            - num_original_sentences (int)
            - detected_topic (int)
            - document_topic_distribution (List[float])
            - topic_keywords (List[str])
            - sentence_details (List[Dict])
            - selected_indices (List[int])
            - summary_sentences (List[str])
            - summary (str)
            - scoring_method (str)
            - similarity_threshold (float)
    """
    # 1. Validation đầu vào
    if vectorizer is None or nmf_model is None:
        raise ValueError("vectorizer và nmf_model không được để trống (None).")

    if not isinstance(text, str) or not text.strip():
        raise ValueError("Văn bản đầu vào không được để trống (text is empty).")

    if num_sentences <= 0:
        raise ValueError(f"num_sentences phải lớn hơn 0, nhận được {num_sentences}.")

    if similarity_threshold < 0.0 or similarity_threshold > 1.0:
        raise ValueError(
            f"similarity_threshold phải nằm trong khoảng [0.0, 1.0], nhận được {similarity_threshold}."
        )

    if scoring_method not in ("dominant", "weighted"):
        raise ValueError(
            f"scoring_method '{scoring_method}' không hợp lệ. Chỉ chấp nhận 'dominant' hoặc 'weighted'."
        )

    # 2. Tách câu nguyên bản
    split_fn = custom_split_sentences if (custom_split_sentences and callable(custom_split_sentences)) else split_sentences
    original_sentences = split_fn(text)

    if not original_sentences:
        raise ValueError("Không tách được câu hợp lệ nào từ văn bản đầu vào.")

    num_total_sentences = len(original_sentences)
    target_num_sentences = min(num_sentences, num_total_sentences)

    # 3. Tiền xử lý phục vụ TF-IDF
    prep_fn = custom_preprocess if (custom_preprocess and callable(custom_preprocess)) else preprocess
    processed_sentences = [prep_fn(s) for s in original_sentences]
    processed_document = " ".join(processed_sentences)

    # 4. TF-IDF transform (KHÔNG FIT LẠI)
    document_vector = vectorizer.transform([processed_document])
    sentence_tfidf_matrix = vectorizer.transform(processed_sentences)

    # 5. Xử lý trường hợp văn bản hoàn toàn ngoài từ vựng TF-IDF (Zero Vector)
    is_zero_doc = False
    if hasattr(document_vector, "nnz"):
        is_zero_doc = (document_vector.nnz == 0)
    else:
        is_zero_doc = bool(np.all(np.asarray(document_vector) == 0))

    n_topics = getattr(nmf_model, "n_components", 0)

    if is_zero_doc:
        selected_indices = list(range(target_num_sentences))
        sentence_details = []
        for i, sent in enumerate(original_sentences):
            sentence_details.append({
                "index": i,
                "sentence": sent,
                "dominant_topic": 0,
                "topic_distribution": [0.0] * n_topics,
                "score": 0.0,
                "selected": i in selected_indices
            })
        summary_sentences = [original_sentences[i] for i in selected_indices]
        summary_text = " ".join(summary_sentences)

        return {
            "original_text": text,
            "sentences": original_sentences,
            "num_original_sentences": num_total_sentences,
            "detected_topic": 0,
            "document_topic_distribution": [0.0] * n_topics,
            "topic_keywords": [],
            "sentence_details": sentence_details,
            "selected_indices": selected_indices,
            "summary_sentences": summary_sentences,
            "summary": summary_text,
            "scoring_method": scoring_method,
            "similarity_threshold": similarity_threshold,
            "warning": "Document contains no known TF-IDF vocabulary.",
            # Tương thích với quy ước chung của Người 4 (evaluation.py & main.py)
            "original": text,
            "topic_id": 0,
            "keywords": [],
            "sentence_scores": [0.0] * num_total_sentences,
        }

    # 6. NMF transform cho document và sentences (KHÔNG TRAIN LẠI)
    doc_topic_raw = nmf_model.transform(document_vector)[0]
    dominant_topic = int(np.argmax(doc_topic_raw)) if np.sum(doc_topic_raw) > 0 else 0
    sentence_topic_matrix = get_sentence_topics(sentence_tfidf_matrix, nmf_model)

    # 7. Tính điểm câu
    scores = score_sentences(
        document_topic_distribution=doc_topic_raw,
        sentence_topic_matrix=sentence_topic_matrix,
        method=scoring_method,
        normalize_topics=normalize_topics
    )

    # 8. Xếp hạng câu theo điểm số giảm dần
    ranked_indices = [int(i) for i in np.argsort(scores)[::-1]]

    # 9. Lọc câu trùng lặp ngữ nghĩa
    selected_ranked = remove_redundancy(
        ranked_indices=ranked_indices,
        sentence_tfidf_matrix=sentence_tfidf_matrix,
        num_sentences=target_num_sentences,
        similarity_threshold=similarity_threshold,
        allow_fill=True,
        sentence_texts=original_sentences
    )

    # 10. BẮT BUỘC: Sắp xếp lại theo thứ tự xuất hiện ban đầu trong văn bản
    selected_indices = sorted(selected_ranked)

    # 11. Trích xuất từ khóa chủ đề (nếu có feature_names)
    resolved_features = feature_names
    if resolved_features is None and hasattr(vectorizer, "get_feature_names_out"):
        try:
            resolved_features = list(vectorizer.get_feature_names_out())
        except Exception:
            resolved_features = None

    topic_keywords: List[str] = []
    if resolved_features is not None and len(resolved_features) > 0:
        if get_top_words is not None:
            try:
                topic_keywords = get_top_words(
                    nmf_model,
                    resolved_features,
                    dominant_topic,
                    top_n=top_n_keywords
                )
            except Exception:
                topic_keywords = []
        else:
            # Fallback lấy trực tiếp từ components_ của NMF
            if hasattr(nmf_model, "components_"):
                topic_weights = nmf_model.components_[dominant_topic]
                top_indices = topic_weights.argsort()[::-1][:top_n_keywords]
                topic_keywords = [resolved_features[i] for i in top_indices if i < len(resolved_features)]

    # 12. Xây dựng danh sách chi tiết từng câu phục vụ phân tích & báo cáo
    sentence_details = []
    for i, sent in enumerate(original_sentences):
        sent_topics = [round(float(v), 6) for v in sentence_topic_matrix[i]]
        dom_topic_sent = int(np.argmax(sentence_topic_matrix[i])) if np.sum(sentence_topic_matrix[i]) > 0 else 0
        sentence_details.append({
            "index": i,
            "sentence": sent,
            "dominant_topic": dom_topic_sent,
            "topic_distribution": sent_topics,
            "score": round(float(scores[i]), 6),
            "selected": (i in selected_indices)
        })

    summary_sentences = [original_sentences[i] for i in selected_indices]
    summary_text = " ".join(summary_sentences)
    sentence_scores_list = [round(float(s), 6) for s in scores]

    return {
        "original_text": text,
        "sentences": original_sentences,
        "num_original_sentences": num_total_sentences,
        "detected_topic": dominant_topic,
        "document_topic_distribution": [round(float(v), 6) for v in doc_topic_raw],
        "topic_keywords": topic_keywords,
        "sentence_details": sentence_details,
        "selected_indices": selected_indices,
        "summary_sentences": summary_sentences,
        "summary": summary_text,
        "scoring_method": scoring_method,
        "similarity_threshold": similarity_threshold,
        # Tương thích với quy ước chung của Người 4 (evaluation.py & main.py)
        "original": text,
        "topic_id": dominant_topic,
        "keywords": topic_keywords,
        "sentence_scores": sentence_scores_list,
    }


# =====================================================================
# STANDALONE TEST CHO NGƯỜI 3
# =====================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("--- KIỂM THỬ ĐỘC LẬP MODULE SUMMARIZER (NGƯỜI 3) ---")
    print("=" * 70)

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import NMF

    # 1. Tạo tập dữ liệu nhỏ đại diện cho 2 chủ đề: Công nghệ & Thể thao
    # Huấn luyện độc lập CHỈ trong khối test, không liên quan đến logic summarize()
    train_corpus = [
        "Trí tuệ nhân tạo và học máy đang làm thay đổi công nghệ sản xuất hiện đại.",
        "Các mô hình AI và phần mềm thông minh giúp tự động hóa phân tích dữ liệu doanh nghiệp.",
        "Hệ thống máy tính và thuật toán số hóa nâng cao năng suất lao động.",
        "Đội tuyển bóng đá quốc gia giành chiến thắng kịch tính trên sân cỏ trong trận chung kết.",
        "Cầu thủ trẻ ghi bàn thắng đẹp mắt đưa đội bóng vô địch giải đấu thể thao quốc tế.",
        "Huấn luyện viên bóng đá chúc mừng các cầu thủ đã thi đấu xuất sắc đầy nỗ lực."
    ]

    print("\n1. Tiền xử lý và huấn luyện vectorizer + NMF demo (k=2)...")
    processed_corpus = [preprocess(doc) for doc in train_corpus]
    demo_vectorizer = TfidfVectorizer()
    X_tfidf = demo_vectorizer.fit_transform(processed_corpus)

    demo_nmf = NMF(n_components=2, random_state=42, init="nndsvda", max_iter=400)
    demo_nmf.fit(X_tfidf)
    print("   -> Huấn luyện TF-IDF & NMF hoàn tất.")

    # 2. Văn bản mới cần tóm tắt (chưa từng xuất hiện trong tập huấn luyện)
    test_document = (
        "Trí tuệ nhân tạo ngày càng được ứng dụng sâu rộng trong các doanh nghiệp hiện đại. "
        "Nhiều công ty lớn áp dụng giải pháp AI và máy học để tối ưu hóa quy trình nghiệp vụ. "
        "Các hệ thống tự động hóa thông minh giúp giải phóng sức lao động của nhân viên. "
        "Tuy nhiên doanh nghiệp cũng cần đặc biệt chú ý đến an toàn và bảo mật dữ liệu. "
        "Việc đào tạo nhân sự sử dụng thành thạo các công cụ số đóng vai trò vô cùng quyết định."
    )

    print("\n2. Thực hiện tóm tắt với 2 phương pháp chấm điểm:")

    for method in ("weighted", "dominant"):
        print(f"\n--- PHƯƠNG PHÁP: {method.upper()} ---")
        result = summarize(
            text=test_document,
            vectorizer=demo_vectorizer,
            nmf_model=demo_nmf,
            num_sentences=2,
            scoring_method=method,
            similarity_threshold=0.7,
            normalize_topics=True
        )

        print(f"Chủ đề phát hiện (dominant_topic): {result['detected_topic']}")
        print(f"Phân phối chủ đề tài liệu: {result['document_topic_distribution']}")
        print(f"Từ khóa tiêu biểu của chủ đề: {result['topic_keywords']}")
        print(f"Các chỉ số câu được chọn: {result['selected_indices']}")
        print("\nChi tiết điểm số từng câu:")
        for detail in result["sentence_details"]:
            sel_mark = "[CHỌN]" if detail["selected"] else "      "
            print(f"  {sel_mark} [{detail['index']}] Điểm: {detail['score']:.4f} | Câu: {detail['sentence']}")

        print(f"\nBản tóm tắt cuối cùng ({method}):")
        print(f'"{result["summary"]}"')

    print("\n" + "=" * 70)
    print(">>> TẤT CẢ CÁC HÀM CỦA NGƯỜI 3 CHẠY HOÀN HẢO! <<<")
    print("=" * 70)
