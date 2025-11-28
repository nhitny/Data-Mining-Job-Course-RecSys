import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sentence_transformers import SentenceTransformer
import os
import logging
from datetime import datetime

# ==============================================================================
# 1. CẤU HÌNH ĐƯỜNG DẪN
# ==============================================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
REPORT_DIR = os.path.join(BASE_DIR, 'reports')

# Tạo các thư mục con
LOG_DIR = os.path.join(REPORT_DIR, 'logs')
TABLE_DIR = os.path.join(REPORT_DIR, 'tables')
FIGURE_DIR = os.path.join(REPORT_DIR, 'figures')

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(TABLE_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)

# Tên 2 file cần so sánh
FILE_NAME_1 = 'courses_processed.csv'   # File của bạn
FILE_NAME_2 = 'coursera_cleaned.csv'    # File của bạn bè

# File log
current_time = datetime.now().strftime("%Y-%m-%d_%H%M%S")
LOG_FILE = os.path.join(LOG_DIR, f'comparison_log_{current_time}.txt')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, mode='w', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

# ==============================================================================
# 2. CÁC HÀM XỬ LÝ
# ==============================================================================

def load_data(filename):
    path = os.path.join(DATA_PROCESSED_DIR, filename)
    if os.path.exists(path):
        try:
            return pd.read_csv(path)
        except Exception as e:
            logging.error(f" [LOAD] Lỗi đọc file '{filename}': {e}")
            return None
    else:
        logging.error(f" [LOAD] Không tìm thấy file '{path}'.")
        return None

def get_text_column(df):
    possible_cols = ['text_all', 'full_content_clean', 'content', 'description']
    for col in possible_cols:
        if col in df.columns: return col
    return None

def analyze_overlap(df, filename):
    logging.info(f"\n--- [1. ANALYSIS] KIỂM TRA ĐA TOPIC: {filename} ---")
    if 'topic' not in df.columns or 'course_name' not in df.columns:
        logging.warning(" File thiếu cột 'topic' hoặc 'course_name'.")
        return

    topic_counts = df.groupby('course_name')['topic'].nunique()
    multi_topic = topic_counts[topic_counts > 1] # > 1 là đa topic
    count = len(multi_topic)
    
    logging.info(f"-> Tổng số khóa học (Unique): {len(topic_counts)}")
    logging.info(f"-> Số khóa học thuộc NHIỀU HƠN 1 Topic: {count}")
    
    if count > 0:
        save_path = os.path.join(TABLE_DIR, f"overlap_analysis_{filename}")
        multi_topic.to_csv(save_path)
        logging.info(f"   (Đã lưu danh sách trùng lặp vào: {save_path})")

def run_clustering_tfidf(df, filename):
    logging.info(f"\n--- [2. ANALYSIS] K-MEANS VỚI TF-IDF: {filename} ---")
    col_text = get_text_column(df)
    if not col_text: return

    k = df['topic'].nunique() if 'topic' in df.columns else 10
    logging.info(f"-> Số cụm K: {k}")

    logging.info(f"-> Chạy TF-IDF trên '{col_text}'...")
    texts = df[col_text].fillna("").astype(str).tolist()
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    matrix = vectorizer.fit_transform(texts)
    
    logging.info("-> Training K-Means...")
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(matrix)
    
    try:
        score = silhouette_score(matrix, labels, sample_size=5000)
        logging.info(f" Silhouette Score (TF-IDF): {score:.4f}")
    except: pass

def run_clustering_bert_and_benchmark(df, filename, model):
    """
    Nhiệm vụ 3 & 4: 
    - Chạy K-Means với SBERT
    - Tính điểm cho Topic gốc (Human Label)
    - So sánh 2 điểm số này
    """
    logging.info(f"\n--- [3 & 4. BENCHMARK] SO SÁNH AI vs HUMAN (SBERT): {filename} ---")
    
    col_text = get_text_column(df)
    
    # 1. Chuẩn bị dữ liệu
    texts = df[col_text].fillna("").astype(str).tolist()
    k = df['topic'].nunique() if 'topic' in df.columns else 10
    
    # 2. Vector hóa SBERT (Load cache hoặc tạo mới)
    vec_filename = filename.replace('.csv', '_vectors.npy')
    vec_path = os.path.join(MODEL_DIR, vec_filename)
    
    if os.path.exists(vec_path):
        logging.info(f"-> Load vector cache ({vec_filename})...")
        matrix = np.load(vec_path)
        if matrix.shape[0] != len(texts): # Re-encode nếu lệch
             matrix = model.encode(texts, show_progress_bar=True)
             np.save(vec_path, matrix)
    else:
        logging.info("-> Encode mới SBERT...")
        matrix = model.encode(texts, show_progress_bar=True)
        if not os.path.exists(MODEL_DIR): os.makedirs(MODEL_DIR)
        np.save(vec_path, matrix)

    # 3. Kịch bản A: AI Phân cụm (K-Means)
    logging.info(f"-> [AI] Training K-Means (K={k})...")
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels_ai = kmeans.fit_predict(matrix)
    
    # 4. Kịch bản B: Con người Phân loại (Topic gốc)
    if 'topic' in df.columns:
        # Chuyển topic text thành số (Label Encoding) để tính toán
        labels_human = df['topic'].astype('category').cat.codes
        has_human_label = True
    else:
        logging.warning(" Không có cột 'topic' -> Không thể tính điểm Human.")
        has_human_label = False

    # 5. Tính điểm và So sánh (Dùng sample_size=10000 cho nhanh và chuẩn)
    logging.info("-> Đang tính toán Silhouette Score...")
    score_ai = silhouette_score(matrix, labels_ai, sample_size=10000)
    
    logging.info(f" Silhouette Score (AI - K-Means):      {score_ai:.4f}")
    
    if has_human_label:
        score_human = silhouette_score(matrix, labels_human, sample_size=10000)
        logging.info(f" Silhouette Score (HUMAN - Topic gốc): {score_human:.4f}")
        
        diff = score_ai - score_human
        logging.info("-" * 40)
        if diff > 0:
            logging.info(f" KẾT LUẬN: AI phân cụm TỐT HƠN Con người (+{diff:.4f})")
        else:
            logging.info(f" KẾT LUẬN: Con người phân loại TỐT HƠN AI ({diff:.4f})")
    
    # Lưu kết quả AI ra file
    df['cluster_sbert'] = labels_ai
    save_path = os.path.join(TABLE_DIR, f"clustered_{filename}")
    cols = [c for c in ['course_name', 'topic', 'cluster_sbert'] if c in df.columns]
    df[cols].to_csv(save_path, index=False)
    logging.info(f" Đã lưu kết quả phân cụm vào: {save_path}")

# ==============================================================================
# 3. CHƯƠNG TRÌNH CHÍNH
# ==============================================================================

def main():
    print(" BẮT ĐẦU SO SÁNH TOÀN DIỆN...")
    logging.info(f"=== PHIÊN CHẠY: {current_time} ===")
    
    cache_path = os.path.join(MODEL_DIR, 'sbert_cache')
    try:
        model = SentenceTransformer('all-MiniLM-L6-v2', cache_folder=cache_path)
    except: return

    # FILE 1
    logging.info(f"\n{'='*10} FILE 1: CỦA BẠN ({FILE_NAME_1}) {'='*10}")
    df1 = load_data(FILE_NAME_1)
    if df1 is not None:
        analyze_overlap(df1, FILE_NAME_1)
        run_clustering_tfidf(df1, FILE_NAME_1)
        run_clustering_bert_and_benchmark(df1, FILE_NAME_1, model) # <--- Hàm mới

    # FILE 2
    logging.info(f"\n{'='*10} FILE 2: CỦA BẠN BÈ ({FILE_NAME_2}) {'='*10}")
    df2 = load_data(FILE_NAME_2)
    if df2 is not None:
        analyze_overlap(df2, FILE_NAME_2)
        run_clustering_tfidf(df2, FILE_NAME_2)
        run_clustering_bert_and_benchmark(df2, FILE_NAME_2, model) # <--- Hàm mới

    print(f"\n Xong! Kiểm tra log tại: {LOG_FILE}")

if __name__ == "__main__":
    main()