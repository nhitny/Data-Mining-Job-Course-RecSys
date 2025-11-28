import pandas as pd
import ast
import re
import os

# --- CẤU HÌNH ĐƯỜNG DẪN ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_DIR = os.path.join(BASE_DIR, 'data', 'raw')
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed')

# Tên file input (Trong folder data/raw)
COURSE_FILE_INPUT = 'coursera_courses.csv' 
JD_FILE_INPUT = 'linkedin_jobs.csv'        

# Tên file output (Sẽ lưu vào data/processed)
COURSE_FILE_OUTPUT = 'courses_processed.csv'
JD_FILE_OUTPUT = 'jd_processed.csv'

def ensure_directory_exists(path):
    if not os.path.exists(path): os.makedirs(path)

def clean_skills(skill_str):
    """Làm sạch cột Skills (list string -> string)"""
    if pd.isna(skill_str): return ""
    try:
        skill_list = ast.literal_eval(skill_str)
        if isinstance(skill_list, list):
            cleaned_list = [re.sub(r'[^\w\s]', '', s).strip() for s in skill_list]
            return " ".join(cleaned_list)
        return str(skill_str)
    except:
        clean_txt = re.sub(r'[\[\]\'\"]', '', str(skill_str))
        return clean_txt.replace(',', ' ')

def text_cleaner(text):
    """Làm sạch văn bản chung"""
    if not isinstance(text, str): return ""
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def clean_numeric_col(value):
    """Chuyển chuỗi số (1.5k, 3m) thành số thực"""
    if pd.isna(value): return 0.0
    val_str = str(value).lower().replace(',', '')
    if 'k' in val_str:
        return float(re.sub(r'[^\d.]', '', val_str)) * 1000
    if 'm' in val_str:
        return float(re.sub(r'[^\d.]', '', val_str)) * 1000000
    try:
        return float(re.sub(r'[^\d.]', '', val_str))
    except:
        return 0.0

def process_coursera_data():
    input_path = os.path.join(RAW_DATA_DIR, COURSE_FILE_INPUT)
    print(f"--- Đang xử lý Coursera Data từ: {input_path} ---")
    
    try:
        # Đọc file với dấu chấm phẩy
        df = pd.read_csv(input_path, sep=';', on_bad_lines='skip', encoding='utf-8')
    except Exception as e:
        print(f"Lỗi đọc file: {e}")
        return

    # --- 1. XỬ LÝ TEXT (CONTENT) ---
    # Skills
    if 'skills_gain' in df.columns:
        df['skills_clean'] = df['skills_gain'].apply(clean_skills)
    else:
        df['skills_clean'] = ""

    # Gộp nội dung để Embed
    cols_to_combine = ['course_name', 'topic', 'skills_clean', 'course_summary', 'what_you_learn']
    
    # Hàm gộp an toàn
    def combine_cols(row):
        content = ""
        for col in cols_to_combine:
            if col in row and pd.notna(row[col]):
                content += " " + str(row[col])
        return content

    df['full_content'] = df.apply(combine_cols, axis=1)
    df['full_content_clean'] = df['full_content'].apply(text_cleaner)

    # --- 2. XỬ LÝ SỐ LIỆU (RANKING) ---
    if 'rating' in df.columns: df['rating'] = df['rating'].apply(clean_numeric_col)
    else: df['rating'] = 0.0
        
    if 'enrolled' in df.columns: df['enrolled'] = df['enrolled'].apply(clean_numeric_col)
    else: df['enrolled'] = 0.0

    if 'review_count' in df.columns: df['review_count'] = df['review_count'].apply(clean_numeric_col)
    else: df['review_count'] = 0.0

    # --- 3. LƯU CÁC CỘT QUAN TRỌNG ---
    ensure_directory_exists(PROCESSED_DATA_DIR)
    output_path = os.path.join(PROCESSED_DATA_DIR, COURSE_FILE_OUTPUT)
    
    # [QUAN TRỌNG] Đã thêm 'topic' vào đây để không bị mất
    keep_cols = [
        'course_name',          # Cần để group
        'topic',                # Cần để đếm số topic và xác định K
        'full_content_clean',   # Cần để vector hóa
        'skills_clean',
        'rating', 'review_count', 'enrolled', 'level', 'duration', 'page_url'
    ]
    
    final_cols = [c for c in keep_cols if c in df.columns]
    
    df[final_cols].to_csv(output_path, index=False)
    print(f"-> Đã lưu file sạch tại: {output_path}")

def process_jd_data():
    input_path = os.path.join(RAW_DATA_DIR, JD_FILE_INPUT)
    print(f"\n--- Đang xử lý JD Data từ: {input_path} ---")
    try:
        df = pd.read_csv(input_path, sep=',', on_bad_lines='skip', encoding='utf-8')
    except: 
        print("Lỗi đọc file JD")
        return
    
    df['full_content'] = (
        df['title'].fillna('') + " " + 
        df['description'].fillna('') + " " +
        df['keyword'].fillna('')
    )
    df['full_content_clean'] = df['full_content'].apply(text_cleaner)

    ensure_directory_exists(PROCESSED_DATA_DIR)
    output_path = os.path.join(PROCESSED_DATA_DIR, JD_FILE_OUTPUT)
    
    cols_to_save = ['title', 'full_content_clean']
    if 'company' in df.columns: cols_to_save.append('company')
    
    df[cols_to_save].to_csv(output_path, index=False)
    print(f"-> Đã lưu file JD sạch tại: {output_path}")

if __name__ == "__main__":
    process_coursera_data()
    process_jd_data()
    print("\n=== HOÀN TẤT BƯỚC 1 (FINAL) ===")