import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os
import sys

class CourseRecommender:
    def __init__(self):
        print("--- KHỞI TẠO HỆ THỐNG GỢI Ý ---")
        
        # 1. TỰ ĐỘNG TÌM ĐƯỜNG DẪN
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # Load file cleaned (ưu tiên)
        self.data_path = os.path.join(self.base_dir, 'data', 'processed', 'coursera_cleaned.csv')
        self.vec_path = os.path.join(self.base_dir, 'models', 'coursera_cleaned_vectors.npy')
        
        if not os.path.exists(self.data_path):
            # Fallback
            self.data_path = os.path.join(self.base_dir, 'data', 'processed', 'courses_processed.csv')
            self.vec_path = os.path.join(self.base_dir, 'models', 'courses_vectors.npy')

        # 2. LOAD DATA
        self.df_courses = pd.read_csv(self.data_path)
        
        # ... (Phần xử lý text/level giữ nguyên) ...
        self.col_content = 'text_all' if 'text_all' in self.df_courses.columns else 'full_content_clean'
        self.df_courses['content_fill'] = self.df_courses[self.col_content].fillna("")
        self.df_courses['course_name'] = self.df_courses['course_name'].fillna("Unknown")
        
        if 'level' in self.df_courses.columns:
            self.df_courses['level_std'] = self.df_courses['level'].astype(str).str.lower()
        else:
            self.df_courses['level_std'] = 'beginner'

        self.df_courses['type_std'] = self.df_courses['course_name'].astype(str).apply(
            lambda x: 'specialization' if 'specialization' in x.lower() or 'certificate' in x.lower() else 'course'
        )
        
        # 3. LOAD VECTOR & MODEL
        self.course_vectors = np.load(self.vec_path)
        
        # Load luôn file CLUSTER đã phân sẵn (quan trọng)
        # File này được tạo ra từ bước visualization.py
        self.cluster_path = self.data_path.replace('.csv', '_clustered.csv')
        if os.path.exists(self.cluster_path):
            df_clustered = pd.read_csv(self.cluster_path)
            # Merge cột cluster_label vào df chính nếu chưa có
            if 'cluster_ai' in df_clustered.columns:
                self.df_courses['cluster_label'] = df_clustered['cluster_ai']
            elif 'cluster_label' in df_clustered.columns:
                self.df_courses['cluster_label'] = df_clustered['cluster_label']
            else:
                self.df_courses['cluster_label'] = -1 # Không có cluster
        else:
            self.df_courses['cluster_label'] = -1

        cache_path = os.path.join(self.base_dir, 'models', 'sbert_cache')
        try:
            self.model = SentenceTransformer('all-MiniLM-L6-v2', cache_folder=cache_path)
        except:
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            
        print("✅ Hệ thống sẵn sàng!")

    def recommend(self, jd_text, user_profile=None, top_k=10, boost_keywords=None):
        # 1. Semantic Search
        jd_vector = self.model.encode([jd_text])
        
        # --- [ỨNG DỤNG K-MEANS VÀO ĐÂY] ---
        # Bước A: Tìm xem JD này giống với các khóa học nào nhất để suy ra Cluster chủ đạo
        # (Thay vì tính toán trung tâm cụm phức tạp, ta lấy cluster của Top 3 khóa giống nhất làm gợi ý)
        preliminary_scores = cosine_similarity(jd_vector, self.course_vectors)[0]
        top_3_indices = preliminary_scores.argsort()[-3:][::-1]
        
        # Lấy cluster xuất hiện nhiều nhất trong Top 3
        top_clusters = [self.df_courses.iloc[i].get('cluster_label', -1) for i in top_3_indices]
        # Tìm cluster phổ biến nhất (Target Cluster)
        target_cluster = max(set(top_clusters), key=top_clusters.count)
        
        # Bước B: Tính toán lại điểm số (Ranking)
        # ... (Phần lấy top 60 giữ nguyên) ...
        top_indices = preliminary_scores.argsort()[-60:][::-1]
        
        candidates = []
        if not user_profile: user_profile = {}
        yoe = user_profile.get('years_experience', 0)
        mode = user_profile.get('learning_mode', 'deep')
        known_skills = [s.lower().strip() for s in user_profile.get('known_skills', [])]
        
        for idx in top_indices:
            score = preliminary_scores[idx]
            course_row = self.df_courses.iloc[idx]
            reasons = []
            
            # --- [LOGIC MỚI] BOOST ĐIỂM CÙNG CLUSTER ---
            current_cluster = course_row.get('cluster_label', -1)
            if current_cluster != -1 and current_cluster == target_cluster:
                score *= 1.2  # Tăng 20% điểm nếu khóa học thuộc cùng nhóm chuyên môn với JD
                # reasons.append("thuộc đúng nhóm chuyên môn") # Có thể hiện hoặc không

            # ... (Các logic Kinh nghiệm, Chế độ học, Skill Gap giữ nguyên) ...
            
            # Logic Kinh nghiệm
            lvl = str(course_row.get('level_std', ''))
            if yoe > 5: # Senior
                if 'beginner' in lvl: score *= 0.6
                elif 'advanced' in lvl: score *= 1.3; reasons.append("hợp Senior")
            elif yoe < 2: # Fresher
                if 'beginner' in lvl: score *= 1.2; reasons.append("tốt cho người mới")

            # Logic Chế độ học
            ctype = str(course_row.get('type_std', ''))
            if mode == 'quick' and ctype == 'course':
                score *= 1.15; reasons.append("ngắn gọn")
            elif mode == 'deep' and ctype == 'specialization':
                score *= 1.2; reasons.append("chuyên sâu")

            # Logic Skill Gap
            cname = str(course_row.get('course_name', '')).lower()
            for skill in known_skills:
                if skill and skill in cname and 'advanced' not in cname:
                    score *= 0.8; break
            
            # Logic Boost Keyword
            content = str(course_row.get('content_fill', '')).lower()
            if boost_keywords:
                for kw in boost_keywords:
                    if kw.lower() in content:
                        score *= 1.4; reasons.append(f"có '{kw}'"); break

            candidates.append({'data': course_row, 'final_score': score, 'reasons': reasons})

        candidates.sort(key=lambda x: x['final_score'], reverse=True)
        return self._generate_path(candidates[:top_k])

    def _generate_path(self, candidates):
        # (Giữ nguyên hàm này)
        path = {"Giai đoạn 1": [], "Giai đoạn 2": [], "Giai đoạn 3": []}
        for item in candidates:
            lvl = str(item['data'].get('level_std', '')).lower()
            ctype = str(item['data'].get('type_std', '')).lower()
            if 'beginner' in lvl: path["Giai đoạn 1"].append(item)
            elif 'advanced' in lvl or 'specialization' in ctype: path["Giai đoạn 3"].append(item)
            else: path["Giai đoạn 2"].append(item)
        for k in path: path[k].sort(key=lambda x: x['final_score'], reverse=True)
        return path