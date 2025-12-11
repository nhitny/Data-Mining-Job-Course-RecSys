from typing import Tuple, Union, Optional
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import sys

# Import wrapper từ file bên cạnh
sys.path.append(os.getcwd()) 
from src.embedding.embedding_model import get_default_model

def _ensure_out_dirs(out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

def _load_courses_arg(courses_arg: Union[pd.DataFrame, str]) -> pd.DataFrame:
    if isinstance(courses_arg, pd.DataFrame):
        return courses_arg.copy()
    if isinstance(courses_arg, str):
        if not os.path.exists(courses_arg):
            raise FileNotFoundError(f"File not found: {courses_arg}")
        return pd.read_csv(courses_arg)
    raise ValueError("Input must be DataFrame or path string")

def build_embeddings(
    courses: Union[pd.DataFrame, str],
    out_dir: str = "data/outputs/embeddings",
    model_name: str = "all-mpnet-base-v2",
    device: Optional[str] = None,
    batch_size: int = 64,
) -> Tuple[str, str]:
    
    _ensure_out_dirs(out_dir)
    df = _load_courses_arg(courses)
    
    print(f"Original Data Rows: {len(df)}")
    
    # 1. Chuẩn hóa tên cột (về chữ thường, bỏ khoảng trắng thừa)
    df.columns = [c.lower().strip() for c in df.columns]
    
    # 2. Xây dựng 'clean_text' từ NHIỀU CỘT (Quan trọng)
    # Dựa trên dataset của bạn, các cột quan trọng là:
    # course_name, topic, course_summary, skills_gain_clean, what_you_learn_clean
    
    # Danh sách các cột ưu tiên để ghép (theo thứ tự quan trọng)
    priority_cols = [
        'course_name', 
        'topic', 
        'course_summary', 
        'what_you_learn_clean', 
        'skills_gain_clean'
    ]
    
    # Chỉ lấy các cột thực sự tồn tại trong file csv
    cols_to_use = [c for c in priority_cols if c in df.columns]
    
    print(f"Constructing embedding text from columns: {cols_to_use}")
    
    # Hàm ghép chuỗi: Fill NaN bằng rỗng -> Ghép lại bằng dấu chấm
    def combine_text(row):
        parts = []
        for col in cols_to_use:
            val = str(row[col])
            # Bỏ qua giá trị nan hoặc rỗng
            if val and val.lower() != 'nan' and val.strip() != '':
                parts.append(val.strip())
        return ". ".join(parts)

    # Áp dụng ghép chuỗi
    # Nếu file csv đã có cột 'text_all' và bạn tin tưởng nó, có thể dùng luôn:
    # if 'text_all' in df.columns:
    #     df['clean_text'] = df['text_all'].fillna("").astype(str)
    # else:
    
    # Tự ghép để kiểm soát chất lượng tốt nhất
    df['clean_text'] = df.apply(combine_text, axis=1)

    # 3. XỬ LÝ DÒNG TRỐNG (Không được xóa dòng nào!)
    # Nếu sau khi ghép mà vẫn rỗng (hiếm), điền placeholder
    empty_mask = df["clean_text"].str.len() < 2 # Nhỏ hơn 2 ký tự coi như rỗng
    empty_count = empty_mask.sum()
    
    if empty_count > 0:
        print(f"Warning: Found {empty_count} rows with empty text. Filling with 'Course Info Unavailable'.")
        df.loc[empty_mask, "clean_text"] = "Course Info Unavailable"
    
    # Reset index để đảm bảo index 0,1,2... khớp tuyệt đối với vector
    df = df.reset_index(drop=True)
    texts = df["clean_text"].tolist()

    print(f"Start embedding ALL {len(texts)} rows...")
    model = get_default_model(model_name=model_name, device=device)
    
    # Encode toàn bộ
    embeddings = model.encode_batch(texts, batch_size=batch_size)

    # --- LƯU FILE ---
    emb_path = os.path.join(out_dir, "course_emb.npy")
    meta_path = os.path.join(out_dir, "course_meta.csv")

    np.save(emb_path, embeddings)
    
    # Lưu metadata (Giữ lại các cột quan trọng để hiển thị UI sau này)
    # Thêm url, rating, duration... từ dataset của bạn
    meta_cols_candidate = [
        'page_url', 'course_name', 'topic', 'rating', 'level', 
        'duration_hours', 'skills_gain_clean', 'clean_text'
    ]
    # Chỉ giữ cột nào có trong df
    keep_cols = [c for c in meta_cols_candidate if c in df.columns]
    
    df_meta = df[keep_cols].copy()
    
    # Đổi tên cột cho thống nhất với hệ thống (nếu cần)
    if 'course_name' in df_meta.columns:
        df_meta = df_meta.rename(columns={'course_name': 'course_name'}) # Giữ nguyên hoặc map sang title
    
    df_meta["emb_index"] = df_meta.index  # Mapping ID dòng với ID vector
    df_meta.to_csv(meta_path, index=False)
    
    return emb_path, meta_path

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--courses", default="data/courses.csv")
    parser.add_argument("--out_dir", default="outputs/embeddings")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    emb, meta = build_embeddings(
        courses=args.courses,
        out_dir=args.out_dir,
        batch_size=args.batch_size
    )
    # Đọc lại file meta để đếm dòng chính xác
    meta_df = pd.read_csv(meta)
    print(f"DONE! Processed {len(meta_df)} rows.")
    print(f"- Embeddings saved to: {emb}")
    print(f"- Metadata saved to:   {meta}")