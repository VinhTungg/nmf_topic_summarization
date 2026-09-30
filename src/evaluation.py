"""
evaluation.py - Đánh giá mô hình chủ đề và chất lượng tóm tắt (Người 4).

Chỉ phụ thuộc numpy + pandas, không cần rouge-score.
Lý do tự cài ROUGE: tokenizer mặc định của thư viện rouge-score chỉ giữ ký tự
a-z/0-9 nên sẽ làm hỏng tiếng Việt có dấu.

Quy ước đầu vào (theo kế hoạch nhóm):
  - Từ topic model (Người 1): X, W, H, (tuỳ chọn) reconstruction error do model báo.
  - Từ summarizer (Người 3): dict có các khóa
        original, summary, topic_id, keywords, sentence_scores, selected_indices
"""

import re
from collections import Counter

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 1. Reconstruction error (NMF)
# ---------------------------------------------------------------------------
def _is_sparse(X):
    return hasattr(X, "tocsr")


def _fro_norm_sq(X):
    """||X||_F^2, dùng được cho cả ma trận dense lẫn sparse."""
    if _is_sparse(X):
        return float(X.multiply(X).sum())
    return float(np.sum(np.asarray(X) ** 2))


def reconstruction_error(X, W, H):
    """Sai số tái tạo Frobenius ||X - WH||_F.

    X có thể là ndarray hoặc scipy sparse (đầu ra TfidfVectorizer). Với sparse,
    dùng khai triển ||X-WH||^2 = ||X||^2 - 2<X, WH> + ||WH||^2 để không phải
    dựng ma trận WH dày (quan trọng khi chạy UIT-ViNews).
    Giá trị này phải khớp với model.reconstruction_err_ của sklearn.NMF.
    """
    W = np.asarray(W)
    H = np.asarray(H)
    if not _is_sparse(X):
        return float(np.linalg.norm(np.asarray(X) - W @ H, "fro"))
    cross = float(((X @ H.T) * W).sum())          # <X, WH>
    wh_sq = float(((W.T @ W) * (H @ H.T)).sum())  # ||WH||^2
    return float(np.sqrt(max(_fro_norm_sq(X) - 2 * cross + wh_sq, 0.0)))


def evaluate_topics(X, W, H, model_error=None):
    """Tổng hợp chỉ số của mô hình chủ đề.

    model_error: sai số do Người 1 báo (model.reconstruction_err_), nếu có,
    dùng để đối chiếu với giá trị tự tính (phát hiện nhầm W/H hoặc sai X).
    """
    err = reconstruction_error(X, W, H)
    x_norm = float(np.sqrt(_fro_norm_sq(X)))
    result = {
        "n_docs": int(X.shape[0]),
        "n_terms": int(X.shape[1]),
        "n_topics": int(np.asarray(H).shape[0]),
        "reconstruction_error": err,
        "relative_error": err / x_norm if x_norm > 0 else float("nan"),
    }
    if model_error is not None:
        result["model_error"] = float(model_error)
        result["matches_model"] = bool(abs(err - model_error) <= 1e-3 * max(1.0, abs(model_error)))
    return result


# ---------------------------------------------------------------------------
# 2. Compression ratio
# ---------------------------------------------------------------------------
def _count_units(text):
    """Đơn vị đo độ dài: số token tách theo khoảng trắng (≈ số âm tiết tiếng Việt)."""
    return len(text.split())


def compression_ratio(original, summary):
    """Tỷ lệ nén = 1 - (độ dài tóm tắt / độ dài gốc) (0-1, càng lớn càng nén mạnh).
    
    Định nghĩa mới giúp giá trị tỷ lệ thuận với mức độ nén.
    """
    n_orig = _count_units(original)
    if n_orig == 0:
        return 0.0
    return 1.0 - (_count_units(summary) / n_orig)


# ---------------------------------------------------------------------------
# 3. ROUGE-1 / ROUGE-2 / ROUGE-L (tự cài, token = âm tiết tiếng Việt)
# ---------------------------------------------------------------------------
def _tokenize(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)  # bỏ dấu câu, giữ chữ có dấu
    return text.split()


def _prf(overlap, ref_total, sys_total):
    p = overlap / sys_total if sys_total else 0.0
    r = overlap / ref_total if ref_total else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return {"precision": p, "recall": r, "f1": f}


def _rouge_n(ref_tokens, sys_tokens, n):
    ref = Counter(tuple(ref_tokens[i:i + n]) for i in range(len(ref_tokens) - n + 1))
    sys_ = Counter(tuple(sys_tokens[i:i + n]) for i in range(len(sys_tokens) - n + 1))
    overlap = sum((ref & sys_).values())
    return _prf(overlap, sum(ref.values()), sum(sys_.values()))


def _lcs_length(a, b):
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b, start=1):
            cur.append(prev[j - 1] + 1 if x == y else max(prev[j], cur[-1]))
        prev = cur
    return prev[-1]


def rouge_score(reference, summary):
    """Trả về {'rouge1': {...}, 'rouge2': {...}, 'rougeL': {...}}, mỗi mục có precision/recall/f1."""
    ref_t, sys_t = _tokenize(reference), _tokenize(summary)
    return {
        "rouge1": _rouge_n(ref_t, sys_t, 1),
        "rouge2": _rouge_n(ref_t, sys_t, 2),
        "rougeL": _prf(_lcs_length(ref_t, sys_t), len(ref_t), len(sys_t)),
    }


# ---------------------------------------------------------------------------
# 4. Đánh giá một bản tóm tắt
# ---------------------------------------------------------------------------
def evaluate_summary(original, summary, reference=None):
    """Chỉ số cho một bản tóm tắt. Có `reference` thì tính thêm ROUGE."""
    metrics = {
        "original_words": _count_units(original),
        "summary_words": _count_units(summary),
        "compression_ratio": compression_ratio(original, summary),
    }
    if reference:
        metrics["rouge"] = rouge_score(reference, summary)
    return metrics


# ---------------------------------------------------------------------------
# 5. Dựng bảng cho báo cáo
# ---------------------------------------------------------------------------
def build_summary_table(results, references=None, doc_ids=None, categories=None, add_mean=True):
    """Bảng: original length, summary length, compression ratio (+ ROUGE F1 nếu có reference)."""
    rows = []
    for i, res in enumerate(results):
        ref = references[i] if references else None
        m = evaluate_summary(res["original"], res["summary"], ref)
        row = {"doc_id": doc_ids[i] if doc_ids else i + 1}
        if categories:
            row["category"] = categories[i]
        row["topic_id"] = res.get("topic_id")
        row["original_words"] = m["original_words"]
        row["summary_words"] = m["summary_words"]
        row["compression_ratio"] = m["compression_ratio"]
        if "rouge" in m:
            row["rouge1_f1"] = m["rouge"]["rouge1"]["f1"]
            row["rouge2_f1"] = m["rouge"]["rouge2"]["f1"]
            row["rougeL_f1"] = m["rouge"]["rougeL"]["f1"]
        rows.append(row)

    df = pd.DataFrame(rows)
    if add_mean and not df.empty:
        metric_cols = [c for c in ["original_words", "summary_words", "compression_ratio",
                                   "rouge1_f1", "rouge2_f1", "rougeL_f1"] if c in df.columns]
        mean_row = {c: None for c in df.columns}
        mean_row["doc_id"] = "Trung binh"
        for c in metric_cols:
            mean_row[c] = df[c].mean()
        df = pd.concat([df, pd.DataFrame([mean_row])], ignore_index=True)
    return df.round(4)


def build_topic_table(top_words):
    """Bảng top keywords theo topic. top_words: {topic_id: [word, ...]}."""
    return pd.DataFrame(
        [{"topic_id": k, "top_keywords": ", ".join(words)} for k, words in top_words.items()]
    )


def build_doc_topic_table(W, doc_ids=None, categories=None):
    """Bảng document-topic (W) + chủ đề chiếm ưu thế. `categories` chỉ để đối chiếu, không dùng để train."""
    W = np.asarray(W)
    df = pd.DataFrame(W, columns=[f"topic_{k}" for k in range(W.shape[1])]).round(4)
    df.insert(0, "doc_id", doc_ids if doc_ids is not None else range(1, len(W) + 1))
    if categories is not None:
        df.insert(1, "category", categories)
    df["dominant_topic"] = W.argmax(axis=1)
    return df


def build_k_table(k_errors):
    """Bảng k - reconstruction error. k_errors: {k: error}."""
    return pd.DataFrame(
        [{"k": k, "reconstruction_error": e} for k, e in sorted(k_errors.items())]
    ).round(4)


def topic_category_crosstab(dominant_topics, categories):
    """Bảng chéo topic x nhãn tham chiếu, dùng để nhận xét thủ công topic có khớp chủ đề thật không."""
    return pd.crosstab(
        pd.Series(categories, name="category"),
        pd.Series(dominant_topics, name="dominant_topic"),
    )
