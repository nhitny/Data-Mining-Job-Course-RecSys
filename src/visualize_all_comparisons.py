import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer
import os

# --- CẤU HÌNH ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
REPORT_DIR = os.path.join(BASE_DIR, 'reports', 'figures')

# Tên file
FILE_CLEANED = 'coursera_cleaned.csv'    # File của bạn bè (Chuẩn hơn)
FILE_PROCESSED = 'courses_processed.csv' # File của bạn

# Model SBERT
CACHE_PATH = os.path.join(MODEL_DIR, 'sbert_cache')
try:
    sbert_model = SentenceTransformer('all-MiniLM-L6-v2', cache_folder=CACHE_PATH)
except:
    sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

def get_text_col(df):
    for col in ['text_all', 'full_content_clean', 'content']:
        if col in df.columns: return col
    return None

def process_tfidf(df):
    """Xử lý TF-IDF: Trả về tọa độ PCA và nhãn cụm"""
    col = get_text_col(df)
    texts = df[col].fillna("").astype(str).tolist()
    k = df['topic'].nunique() if 'topic' in df.columns else 10
    
    # 1. Vector hóa
    tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
    matrix = tfidf.fit_transform(texts)
    
    # 2. K-Means
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(matrix)
    
    # 3. PCA
    pca = PCA(n_components=2)
    coords = pca.fit_transform(matrix.toarray())
    
    return coords, clusters, k

def process_sbert(df, filename):
    """Xử lý SBERT: Trả về tọa độ PCA và nhãn cụm"""
    col = get_text_col(df)
    texts = df[col].fillna("").astype(str).tolist()
    k = df['topic'].nunique() if 'topic' in df.columns else 10
    
    # 1. Load/Create Vector
    vec_name = filename.replace('.csv', '_vectors.npy')
    vec_path = os.path.join(MODEL_DIR, vec_name)
    
    if os.path.exists(vec_path):
        matrix = np.load(vec_path)
        if matrix.shape[0] != len(texts):
            matrix = sbert_model.encode(texts, show_progress_bar=True)
    else:
        print(f"   (Đang encode mới cho {filename}...)")
        matrix = sbert_model.encode(texts, show_progress_bar=True)
        # np.save(vec_path, matrix) # Có thể save lại nếu muốn

    # 2. K-Means
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(matrix)
    
    # 3. PCA
    pca = PCA(n_components=2)
    coords = pca.fit_transform(matrix)
    
    return coords, clusters, k

def draw_chart(ax, x, y, hue, title, palette='tab10'):
    sns.scatterplot(x=x, y=y, hue=hue, palette=palette, s=15, alpha=0.6, ax=ax, legend=False)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")

def main():
    os.makedirs(REPORT_DIR, exist_ok=True)
    
    # Load Data
    print("-> Loading Data...")
    df_clean = pd.read_csv(os.path.join(DATA_DIR, FILE_CLEANED))
    df_proc = pd.read_csv(os.path.join(DATA_DIR, FILE_PROCESSED))
    
    # =========================================================
    # ẢNH 1: SO SÁNH VỀ TF-IDF
    # =========================================================
    print("\n Đang vẽ Ảnh 1: Phân tích TF-IDF...")
    fig1, axes1 = plt.subplots(1, 3, figsize=(24, 7))
    
    # Lấy dữ liệu TF-IDF
    coords_clean, clust_clean, k_clean = process_tfidf(df_clean)
    coords_proc, clust_proc, k_proc = process_tfidf(df_proc)
    
    # 1.1: Topic Gốc (Dựa trên nền Cleaned Data)
    draw_chart(axes1[0], coords_clean[:,0], coords_clean[:,1], df_clean['topic'], 
               f"PHÂN LOẠI GỐC (Topic)\n(Trên nền dữ liệu Cleaned)", palette='tab20')
    
    # 1.2: TF-IDF trên Data Cleaned
    draw_chart(axes1[1], coords_clean[:,0], coords_clean[:,1], clust_clean, 
               f"K-MEANS (TF-IDF)\nData: {FILE_CLEANED}", palette='viridis')
    
    # 1.3: TF-IDF trên Data Processed
    draw_chart(axes1[2], coords_proc[:,0], coords_proc[:,1], clust_proc, 
               f"K-MEANS (TF-IDF)\nData: {FILE_PROCESSED}", palette='viridis')
    
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, 'compare_1_TFIDF.png'), dpi=300)
    print(" Đã lưu: compare_1_TFIDF.png")

    # =========================================================
    # ẢNH 2: SO SÁNH VỀ SBERT
    # =========================================================
    print("\n Đang vẽ Ảnh 2: Phân tích SBERT...")
    fig2, axes2 = plt.subplots(1, 3, figsize=(24, 7))
    
    # Lấy dữ liệu SBERT
    coords_clean_sb, clust_clean_sb, _ = process_sbert(df_clean, FILE_CLEANED)
    coords_proc_sb, clust_proc_sb, _ = process_sbert(df_proc, FILE_PROCESSED)
    
    # 2.1: Topic Gốc (Dựa trên nền SBERT Cleaned)
    draw_chart(axes2[0], coords_clean_sb[:,0], coords_clean_sb[:,1], df_clean['topic'], 
               f"PHÂN LOẠI GỐC (Topic)\n(Trên không gian Vector SBERT)", palette='tab20')
    
    # 2.2: SBERT trên Data Cleaned
    draw_chart(axes2[1], coords_clean_sb[:,0], coords_clean_sb[:,1], clust_clean_sb, 
               f"K-MEANS (SBERT)\nData: {FILE_CLEANED}", palette='Spectral')
    
    # 2.3: SBERT trên Data Processed
    draw_chart(axes2[2], coords_proc_sb[:,0], coords_proc_sb[:,1], clust_proc_sb, 
               f"K-MEANS (SBERT)\nData: {FILE_PROCESSED}", palette='Spectral')
    
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, 'compare_2_SBERT.png'), dpi=300)
    print(" Đã lưu: compare_2_SBERT.png")

if __name__ == "__main__":
    main()