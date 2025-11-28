import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import ndcg_score
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os
import sys

# --- CẤU HÌNH ĐƯỜNG DẪN ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_FILE = os.path.join(BASE_DIR, 'data', 'ground_truth.csv')
REPORT_IMG = os.path.join(BASE_DIR, 'reports', 'figures', 'evaluation_metrics.png')
MODEL_DIR = os.path.join(BASE_DIR, 'models')

# Tạo thư mục figures nếu chưa có
os.makedirs(os.path.dirname(REPORT_IMG), exist_ok=True)

# Load SBERT để tính Diversity (Độ đa dạng)
CACHE_PATH = os.path.join(MODEL_DIR, 'sbert_cache')
try:
    print(" Đang load model để tính Diversity...")
    model = SentenceTransformer('all-MiniLM-L6-v2', cache_folder=CACHE_PATH)
except:
    model = SentenceTransformer('all-MiniLM-L6-v2')

def calculate_diversity(course_list):
    """
    Tính độ đa dạng: 1 - Độ tương đồng trung bình giữa các khóa học
    """
    if len(course_list) < 2: return 0
    
    # Vector hóa nội dung các khóa học
    vecs = model.encode(course_list, show_progress_bar=False)
    
    # Tính ma trận tương đồng
    sim_matrix = cosine_similarity(vecs)
    
    # Chỉ lấy tam giác trên của ma trận (trừ đường chéo)
    n = len(course_list)
    triu_indices = np.triu_indices(n, k=1)
    similarities = sim_matrix[triu_indices]
    
    avg_sim = np.mean(similarities)
    return 1 - avg_sim # Càng khác nhau thì Diversity càng cao

def evaluate_system():
    print("\n--- BẮT ĐẦU ĐÁNH GIÁ CHUYÊN SÂU (FULL METRICS) ---")
    
    if not os.path.exists(INPUT_FILE):
        print(f" Lỗi: Không tìm thấy file {INPUT_FILE}")
        print(" Hãy chạy 'python src/data_labeling.py' trước!")
        return

    df = pd.read_csv(INPUT_FILE)
    
    # Biến lưu kết quả từng JD
    metrics_per_jd = []
    
    # Nhóm theo JD để tính toán Ranking Metrics
    grouped = df.groupby('jd_title')
    
    print(f"-> Đang phân tích kết quả trên {len(grouped)} JD mẫu...")
    
    for jd_title, group in grouped:
        # Sắp xếp theo điểm hệ thống giảm dần (Mô phỏng thứ tự hiển thị Top 1 -> Top 5)
        group = group.sort_values('system_score', ascending=False)
        
        # Lấy nhãn thực tế (Gemini chấm)
        y_true = group['human_label'].tolist()
        # Lấy điểm hệ thống
        y_score = group['system_score'].tolist()
        
        # 1. Tính Precision (Tỷ lệ đúng trong danh sách này)
        precision = np.mean(y_true)
        
        # 2. Tính MRR (Vị trí đúng đầu tiên)
        try:
            first_correct_idx = y_true.index(1)
            mrr = 1 / (first_correct_idx + 1)
        except ValueError:
            mrr = 0 # Không có cái nào đúng
            
        # 3. Tính NDCG (Chất lượng xếp hạng)
        # Scikit-learn yêu cầu shape (1, n_samples)
        if len(y_true) > 1:
            ndcg = ndcg_score([y_true], [y_score])
        else:
            ndcg = y_true[0] # Nếu chỉ có 1 item
            
        # 4. Tính Diversity
        # Cần lấy nội dung khóa học. Nếu file ground_truth thiếu cột này thì bỏ qua
        diversity = 0
        if 'course_name' in group.columns:
            # Dùng tên khóa học để tính độ đa dạng (nhanh hơn dùng full content)
            diversity = calculate_diversity(group['course_name'].tolist())
            
        metrics_per_jd.append({
            'Precision': precision,
            'MRR': mrr,
            'NDCG': ndcg,
            'Diversity': diversity
        })

    # --- TỔNG HỢP KẾT QUẢ ---
    df_metrics = pd.DataFrame(metrics_per_jd)
    final_metrics = df_metrics.mean()
    
    print("\n" + "="*50)
    print(" BÁO CÁO HIỆU NĂNG HỆ THỐNG (FINAL REPORT)")
    print("="*50)
    
    for metric, value in final_metrics.items():
        print(f" {metric:<10}: {value:.4f} \t({value*100:.1f}%)")
        
    print("-" * 50)
    print(" Giải thích ý nghĩa:")
    print(" - Precision: Tỷ lệ khóa học hữu ích.")
    print(" - MRR:       Khóa học đúng thường nằm ở top đầu.")
    print(" - NDCG:      Thứ tự xếp hạng chuẩn xác (Cái tốt nhất lên đầu).")
    print(" - Diversity: Danh sách gợi ý đa dạng, không bị trùng lặp.")
    print("=" * 50)

    # --- VẼ BIỂU ĐỒ ---
    print("-> Đang vẽ biểu đồ...")
    plt.figure(figsize=(10, 6))
    
    # Chọn màu đẹp
    colors = ['#2ecc71', '#3498db', '#9b59b6', '#f1c40f']
    ax = sns.barplot(x=final_metrics.index, y=final_metrics.values, palette=colors, edgecolor='black')
    
    plt.ylim(0, 1.15)
    plt.title('Evaluation Metrics (Đánh giá bởi Gemini)', fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('Score (0-1)', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    
    # Hiển thị số trên cột
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.2%}', 
                   (p.get_x() + p.get_width() / 2., p.get_height()), 
                   ha = 'center', va = 'center', 
                   xytext = (0, 9), 
                   textcoords = 'offset points',
                   fontweight='bold', fontsize=11)

    plt.tight_layout()
    plt.savefig(REPORT_IMG, dpi=300)
    print(f" Đã lưu biểu đồ báo cáo tại: {REPORT_IMG}")

if __name__ == "__main__":
    evaluate_system()