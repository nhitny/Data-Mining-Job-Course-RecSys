import pandas as pd
import os

class DataLoader:
    def __init__(self, data_dir="data"):
        # Đường dẫn tuyệt đối hoặc tương đối tới thư mục data
        self.data_dir = data_dir
        self.courses_path = os.path.join(data_dir, "courses.csv")
        self.jds_path = os.path.join(data_dir, "jds.csv")

    def load_courses(self):
        if not os.path.exists(self.courses_path):
            raise FileNotFoundError(f"Không tìm thấy file: {self.courses_path}")
        
        df = pd.read_csv(self.courses_path)
        print(f"Loaded Courses: {df.shape[0]} rows, {df.shape[1]} columns.")
        # Fill NaN để tránh lỗi khi embed
        df = df.fillna("") 
        return df

    def load_jds(self):
        if not os.path.exists(self.jds_path):
            raise FileNotFoundError(f"Không tìm thấy file: {self.jds_path}")
            
        df = pd.read_csv(self.jds_path)
        print(f"Loaded JDs: {df.shape[0]} rows.")
        df = df.fillna("")
        return df

# Test nhanh nếu chạy trực tiếp file này
if __name__ == "__main__":
    loader = DataLoader()
    loader.load_courses()
    loader.load_jds()