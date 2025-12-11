import os
import sys
import yaml
import json
import itertools
import numpy as np
import time
from tqdm import tqdm

# --- CẤU HÌNH ĐƯỜNG DẪN ---
current_file = os.path.abspath(__file__)
src_dir = os.path.dirname(os.path.dirname(current_file)) # .../src
project_root = os.path.dirname(src_dir)                  # .../data_mining_project
if src_dir not in sys.path: sys.path.insert(0, src_dir)
if project_root not in sys.path: sys.path.insert(0, project_root)

# Import hệ thống chính và module tính điểm
from src.main import CourseRecommenderSystem
from src.recommender.scoring import Scorer

def calculate_metrics(predictions, ground_truth, k=5):
    """
    Tính Precision@K: Tỷ lệ khóa học đúng trong top K gợi ý
    """
    if not ground_truth: return 0.0
    
    # Chỉ lấy ID của top K
    pred_ids = [p['id'] for p in predictions[:k]]
    
    # Tìm giao thoa giữa gợi ý và đáp án đúng
    hits = len(set(pred_ids).intersection(set(ground_truth)))
    
    return hits / k

class WeightSearcher:
    def __init__(self, config_path="configs/grid_search.yaml"):
        self.base_dir = project_root
        self.config_path = os.path.join(self.base_dir, config_path)
        
        # 1. Load Config
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Config not found at {self.config_path}")
            
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        # 2. Load System (Chỉ load 1 lần để tiết kiệm RAM/CPU)
        print(">>> Loading Recommender System Core...")
        self.system = CourseRecommenderSystem(base_dir=self.base_dir)
        
        # 3. Load Test Data (Ground Truth)
        test_path = os.path.join(self.base_dir, self.config['grid_search']['evaluation']['test_data_path'])
        if not os.path.exists(test_path):
            raise FileNotFoundError(f"Test data not found at {test_path}. Run create_dummy_testset.py first!")
            
        with open(test_path, 'r') as f:
            self.test_data = json.load(f)
        print(f" Loaded {len(self.test_data)} test cases.")

    def generate_grid(self):
        """Tạo lưới tổ hợp các tham số từ file yaml"""
        params = self.config['grid_search']['weights']
        keys = params.keys()
        values = params.values()
        
        # Tạo tất cả các tổ hợp (Cartesian Product)
        grid = []
        for combo in itertools.product(*values):
            weight_dict = dict(zip(keys, combo))
            
            # Logic lọc bớt: Semantic + BM25 nên xấp xỉ 1.0 (0.9 - 1.1)
            # Để tránh việc các trọng số quá nhỏ hoặc quá lớn làm lệch scale
            total_base = weight_dict['semantic'] + weight_dict['bm25']
            if 0.9 <= total_base <= 1.1:
                grid.append(weight_dict)
                
        print(f"Generated {len(grid)} weight combinations to test.")
        return grid

    def run(self):
        grid = self.generate_grid()
        top_k = self.config['grid_search']['evaluation']['top_k']
        
        best_score = -1.0
        best_config = None
        
        print("\n>>> STARTING GRID SEARCH...")
        start_time = time.time()
        
        # Duyệt qua từng bộ trọng số (sử dụng tqdm để hiện thanh tiến trình)
        for i, weights in enumerate(tqdm(grid, desc="Optimizing")):
            
            # CẬP NHẬT TRỌNG SỐ CHO HỆ THỐNG
            # Chúng ta thay nóng đối tượng scorer bên trong hệ thống
            self.system.scorer = Scorer(weights=weights)
            
            avg_precision = 0.0
            
            # Chạy đánh giá trên toàn bộ tập test
            for case in self.test_data:
                jd = case['jd_text']
                true_ids = case['correct_course_ids']
                
                # Gọi hàm recommend (Lưu ý: recommend trong main.py trả về dict)
                # Ta gọi trực tiếp hàm xử lý để nhanh hơn nếu muốn, nhưng gọi qua recommend cho an toàn
                # Để log không bị rác, ta có thể tạm tắt print trong main (optional)
                
                # Chạy gợi ý
                # Lưu ý: Hàm recommend của main.py có in log, nếu chạy nhiều sẽ rất rối.
                # Ở đây ta chấp nhận log hoặc sửa main.py để có chế độ silent.
                # Cách nhanh nhất: Tạm thời chuyển hướng stdout nếu cần, hoặc cứ để nó in.
                result = self.system.recommend(jd, top_k=top_k)
                
                # Tính điểm
                score = calculate_metrics(result['recommendations'], true_ids, k=top_k)
                avg_precision += score
            
            # Trung bình Precision cho bộ trọng số này
            avg_precision /= len(self.test_data)
            
            # Kiểm tra nếu tốt hơn kỷ lục cũ
            if avg_precision > best_score:
                best_score = avg_precision
                best_config = weights
                # tqdm.write giúp in ra màn hình mà không phá vỡ thanh tiến trình
                tqdm.write(f"New Best Found! P@{top_k}={best_score:.4f} | Weights: {weights}")

        total_time = time.time() - start_time
        
        # KẾT QUẢ CUỐI CÙNG
        print("\n" + "="*60)
        print("GRID SEARCH COMPLETED")
        print("="*60)
        print(f"Time Taken: {total_time:.2f}s")
        print(f"Best Precision@{top_k}: {best_score:.4f}")
        print("Best Configuration:")
        print(json.dumps(best_config, indent=4))
        
        # Lưu kết quả
        out_path = os.path.join(self.base_dir, self.config['grid_search']['evaluation']['output_path'])
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, 'w') as f:
            json.dump(best_config, f, indent=4)
        print(f"\nSaved best weights to: {out_path}")

if __name__ == "__main__":
    try:
        searcher = WeightSearcher()
        searcher.run()
    except KeyboardInterrupt:
        print("\nSearch interrupted by user.")