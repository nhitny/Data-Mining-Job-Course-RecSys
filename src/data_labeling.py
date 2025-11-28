import pandas as pd
import google.generativeai as genai
import os
import time
import random
import sys

import pandas as pd
import google.generativeai as genai
import os
import time
import random
import sys
from dotenv import load_dotenv
load_dotenv()

# --- FIX IMPORT ---
# Thêm đường dẫn src để Python tìm thấy file recommendation_path.py
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Sửa tên file import cho đúng với file hiện tại của bạn
try:
    from recommendation_path import CourseRecommender 
except ImportError:
    print(" Không tìm thấy 'recommendation_path.py'. Đang thử tìm 'recommendation_v3.py'...")
    try:
        from recommendation_path import CourseRecommender
    except:
        print(" Lỗi: Không tìm thấy file code gợi ý nào cả!")
        exit()

# --- CẤU HÌNH ---
# --- CẤU HÌNH ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed')
OUTPUT_FILE = os.path.join(BASE_DIR, 'data', 'ground_truth.csv')

# Load API key từ biến môi trường
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError(
        "GOOGLE_API_KEY is not set. "
        "Hãy set biến môi trường GOOGLE_API_KEY."
    )

# Cấu hình Gemini
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-2.0-flash')

def ask_gemini_safe(jd_text, course_name, course_content):
    """
    Hàm gửi prompt có cơ chế "Thử lại" nếu bị lỗi 429 (Rate Limit)
    """
    prompt = f"""
    Bạn là chuyên gia tuyển dụng. Hãy đánh giá độ phù hợp:
    
    JOB: {jd_text[:800]}
    COURSE: {course_name} - {course_content[:400]}
    
    Yêu cầu: Trả về 1 nếu Hữu ích/Liên quan. Trả về 0 nếu Không liên quan.
    Chỉ trả về 1 số duy nhất.
    """
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = model.generate_content(prompt)
            result = response.text.strip()
            if '1' in result: return 1
            return 0
            
        except Exception as e:
            error_msg = str(e)
            # Nếu lỗi 429 (Hết quota) -> Đợi lâu hơn (20s)
            if "429" in error_msg or "quota" in error_msg.lower():
                print(f" Đang bị Google chặn (Rate Limit). Đợi 20s rồi thử lại lần {attempt+1}...")
                time.sleep(20) 
            else:
                print(f"  Lỗi khác: {e}")
                return 0 
                
    return 0 

def create_dataset():
    print("--- BẮT ĐẦU TẠO DỮ LIỆU LABEL (SAFE MODE) ---")
    
    try:
        recsys = CourseRecommender()
    except Exception as e:
        print(f" Lỗi khởi tạo hệ thống: {e}")
        return
    
    # Ưu tiên tìm file jds_final (chuẩn), nếu không có thì tìm jd_processed
    jd_path = os.path.join(DATA_DIR, 'jds_final.csv')
    if not os.path.exists(jd_path):
        jd_path = os.path.join(DATA_DIR, 'jd_processed.csv')

    if not os.path.exists(jd_path):
        print(" Không tìm thấy file JD nào cả.")
        return

    df_jd = pd.read_csv(jd_path)
    # Lấy 20 JD ngẫu nhiên
    sample_jds = df_jd.sample(n=20, random_state=42)
    
    labeled_data = []
    
    print(f"-> Đang chấm điểm cho 20 JD mẫu...")
    
    for i, (_, row_jd) in enumerate(sample_jds.iterrows()):
        # Lấy nội dung JD an toàn (tìm các tên cột có thể xảy ra)
        jd_text = row_jd.get('content', row_jd.get('full_content_clean', ''))
        jd_title = row_jd.get('title', 'Job')
        
        print(f"\nProcessing [{i+1}/20]: {jd_title}...")
        
        # Lấy Top 5 khóa học
        user_profile = {'years_experience': 0, 'learning_mode': 'deep'}
        path_dict = recsys.recommend(jd_text, user_profile, top_k=5)
        
        candidates = []
        for stage in path_dict.values():
            candidates.extend(stage)
            
        for item in candidates:
            course = item['data']
            c_name = course['course_name']
            
            # [FIX LỖI QUAN TRỌNG] Lấy nội dung khóa học an toàn
            # Vì file của bạn bè dùng 'text_all', file của bạn dùng 'full_content_clean'
            # Code này sẽ tự động tìm cái nào có thì lấy
            c_content = course.get('text_all', course.get('full_content_clean', ''))
            
            # Gọi hàm an toàn mới
            label = ask_gemini_safe(jd_text, c_name, str(c_content))
            
            print(f"   -> Gemini chấm: {label} | Course: {c_name[:40]}...")
            
            labeled_data.append({
                'jd_title': jd_title,
                'course_name': c_name,
                'human_label': label,
                'system_score': item['final_score'] # Lưu thêm điểm hệ thống để tính NDCG sau này
            })
            
            # Nghỉ 5 giây
            time.sleep(5) 

    df_result = pd.DataFrame(labeled_data)
    df_result.to_csv(OUTPUT_FILE, index=False)
    print(f"\n=== XONG! Đã lưu file label tại: {OUTPUT_FILE} ===")
    
    if not df_result.empty:
        accuracy = df_result['human_label'].mean() * 100
        print(f" Độ chính xác trung bình: {accuracy:.1f}%")

if __name__ == "__main__":
    create_dataset()