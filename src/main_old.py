# import os
# import sys
# import pandas as pd
# import numpy as np
# import time
# import argparse
# import json

# # --- CẤU HÌNH ĐƯỜNG DẪN ---
# current_file = os.path.abspath(__file__)
# src_dir = os.path.dirname(current_file)
# project_root = os.path.dirname(src_dir)
# if src_dir not in sys.path: sys.path.insert(0, src_dir)
# if project_root not in sys.path: sys.path.insert(0, project_root)

# # --- IMPORT AN TOÀN ---
# def safe_import():
#     global JDCleaner, ExperienceMapper, SkillExtractor, JDSummarizer
#     global EmbeddingModel, ClusterAssigner
#     global SemanticSearcher, BM25Searcher, CandidateUnion
#     global Scorer, BusinessRules, Explainability
    
#     try:
#         from src.jd_processing.jd_cleaner import JDCleaner
#         from src.jd_processing.experience_mapper import ExperienceMapper
#         from src.jd_processing.skill_extractor import SkillExtractor
#         from src.jd_processing.jd_summarizer import JDSummarizer
#         from src.embedding.embedding_model import EmbeddingModel
#         from src.clustering.cluster_assigner import ClusterAssigner
#         from src.retrieval.semantic_search import SemanticSearcher
#         from src.retrieval.bm25_retriever import BM25Searcher
#         from src.retrieval.candidate_union import CandidateUnion
#         from src.recommender.scoring import Scorer
#         from src.recommender.rules import BusinessRules       
#         from src.recommender.explainability import Explainability 
#     except ImportError:
#         # Fallback
#         from jd_processing.jd_cleaner import JDCleaner
#         from jd_processing.experience_mapper import ExperienceMapper
#         from jd_processing.skill_extractor import SkillExtractor
#         from jd_processing.jd_summarizer import JDSummarizer
#         from embedding.embedding_model import EmbeddingModel
#         from clustering.cluster_assigner import ClusterAssigner
#         from retrieval.semantic_search import SemanticSearcher
#         from retrieval.bm25_retriever import BM25Searcher
#         from retrieval.candidate_union import CandidateUnion
#         from recommender.scoring import Scorer
#         from recommender.rules import BusinessRules
#         from recommender.explainability import Explainability

# safe_import()

# class CourseRecommenderSystem:
#     def __init__(self, base_dir=None):
#         self.base_dir = base_dir if base_dir else project_root
#         print("="*60)
#         print(f">>> SYSTEM INITIALIZED (Root: {self.base_dir})")
#         print("="*60)
        
#         self.paths = {
#             "meta": f"{self.base_dir}/outputs/embeddings/course_meta.csv",
#             "faiss": f"{self.base_dir}/outputs/indices/faiss_index.bin",
#             "bm25": f"{self.base_dir}/outputs/models/bm25_model.pkl",
#             "pca": f"{self.base_dir}/outputs/models/pca_model.pkl",
#             "kmeans": f"{self.base_dir}/outputs/models/kmeans_model.pkl",
#             "summ_model": f"{self.base_dir}/models",
#             "best_weights": f"{self.base_dir}/outputs/models/best_weights.json"
#         }
        
#         # 1. Load Metadata
#         if os.path.exists(self.paths["meta"]):
#             self.course_df = pd.read_csv(self.paths["meta"])
#             if 'emb_index' not in self.course_df.columns:
#                 self.course_df['emb_index'] = self.course_df.index
#             self.course_lookup = self.course_df.set_index('emb_index').to_dict('index')
#             print(f"Metadata: {len(self.course_df)} courses")
#         else:
#             print("Meta file missing")

#         # 2. Load Processors
#         print("\n[1/4] Loading Processors...")
#         self.cleaner = JDCleaner()
#         self.exp_mapper = ExperienceMapper()
#         self.skill_extractor = SkillExtractor(self.paths["meta"])
#         self.summarizer = JDSummarizer(model_dir=self.paths["summ_model"])

#         # 3. Load AI Models
#         print("\n[2/4] Loading AI Models...")
#         self.embedder = EmbeddingModel()
#         self.cluster_assigner = ClusterAssigner(self.paths["pca"], self.paths["kmeans"])

#         # 4. Load Search Engines
#         print("\n[3/4] Loading Retrieval Engines...")
#         self.semantic_searcher = SemanticSearcher(self.paths["faiss"])
#         self.bm25_searcher = BM25Searcher(model_path=self.paths["bm25"])
#         self.retriever = CandidateUnion(self.semantic_searcher, self.bm25_searcher)
        
#         # 5. Load Logic Modules (Rules & Explainability)
#         print("\n[4/4] Loading Logic & Rules...")
#         self.rules = BusinessRules()
#         self.explainer = Explainability()
        
#         # Load weights tối ưu nếu có (từ weight_search.py)
#         weights = None
#         if os.path.exists(self.paths["best_weights"]):
#             with open(self.paths["best_weights"], 'r') as f:
#                 weights = json.load(f)
#             print("   -> Loaded Optimized Weights from Grid Search!")
#         self.scorer = Scorer(weights=weights)
        
#         print("\nREADY!\n")

#     # def recommend(self, raw_jd, top_k=5):
#     #     t0 = time.time()
        
#     #     # --- B1: PHÂN TÍCH JD ---
#     #     clean_text = self.cleaner.clean(raw_jd)
#     #     summary = self.summarizer.summarize(raw_jd)
#     #     exp_info = self.exp_mapper.map_experience(clean_text)
#     #     skills = self.skill_extractor.extract(clean_text)
        
#     #     user_profile = {
#     #         "level": exp_info['level'],
#     #         "years_exp": exp_info['years'],
#     #         "skills": skills,
#     #         "domain": "IT", # Có thể trích xuất thêm
#     #         "title": raw_jd[:50]
#     #     }
#     #     print(f">>> PROFILE: {user_profile['level']} | Skills: {skills[:5]}...")

#     #     # --- B2: CLUSTERING ---
#     #     jd_vector = self.embedder.get_embedding(clean_text)
#     #     jd_cluster_id = self.cluster_assigner.get_cluster(jd_vector.reshape(1, -1))
#     #     print(f">>> CLUSTER: Assigned to Group #{jd_cluster_id}")

#     #     # --- B3: RETRIEVAL ---
#     #     candidates = self.retriever.get_candidates(clean_text, jd_vector, top_k_each=50)
        
#     #     # --- B4: SCORING & RULES (QUAN TRỌNG) ---
#     #     ranked_results = []
#     #     for cand in candidates:
#     #         c_id = cand['id']
#     #         if c_id not in self.course_lookup: continue
            
#     #         # Gộp thông tin từ DB vào candidate
#     #         info = self.course_lookup[c_id]
#     #         full_cand = {**cand, **info}
            
#     #         # 1. Áp dụng Business Rule (Nhân hệ số phạt/thưởng)
#     #         rule_multiplier = self.rules.apply_rules(user_profile, full_cand)
            
#     #         # 2. Tính điểm qua Scorer (Semantic + Skill + Cluster)
#     #         # Lưu ý: Cần chỉnh Scorer một chút để nhận rule_multiplier nếu chưa có
#     #         # Ở đây ta nhân thủ công
#     #         base_score, breakdown = self.scorer.calculate_score(full_cand, user_profile, jd_cluster_id)
            
#     #         # 3. Điểm cuối cùng = Điểm thuật toán * Điểm luật
#     #         final_score = base_score * rule_multiplier
            
#     #         # Cập nhật breakdown để giải thích
#     #         breakdown['rule_multiplier'] = rule_multiplier
#     #         breakdown['final_score'] = final_score
            
#     #         # 4. Tạo giải thích tự động (Natural Language)
#     #         explanation_json = self.explainer.generate_explanation(info.get('course_name'), breakdown)
            
#     #         ranked_results.append({
#     #             "id": c_id,
#     #             "title": info.get('course_name'),
#     #             "score": final_score,
#     #             "level": info.get('level'),
#     #             "url": info.get('page_url'),
#     #             "breakdown": breakdown,
#     #             "explanation": explanation_json
#     #         })

#     #     # Sắp xếp & Cắt Top K
#     #     ranked_results.sort(key=lambda x: x['score'], reverse=True)
#     #     final_recs = ranked_results[:top_k]
        
#     #     return {
#     #         "time": f"{time.time()-t0:.2f}s",
#     #         "summary": summary,
#     #         "profile": user_profile,
#     #         "recommendations": final_recs
#     #     }
#     def recommend(self, raw_jd, user_years=None, top_k=10):
#         t0 = time.time()

#         # ==============================
#         # B1. PHÂN TÍCH JD
#         # ==============================

#         clean_text = self.cleaner.clean(raw_jd)

#         # [THAY ĐỔI QUAN TRỌNG 1] 
#         # Dùng LLM để tóm tắt JD → summary dùng cho embedding semantic
#         summary = self.summarizer.summarize(clean_text)

#         # Làm sạch summary lại lần nữa để tránh ký tự thừa
#         clean_summary = self.cleaner.clean(summary)

#         exp_info = self.exp_mapper.map_experience(clean_text)
#         skills = self.skill_extractor.extract(clean_text)

#         user_profile = {
#             "level": exp_info['level'],
#             "years_exp": exp_info['years'],
#             "skills": skills,
#             "domain": "IT",
#             "title": raw_jd[:50],
#             "summary": summary,
#             "user_years": user_years
#         }

#         print(f">>> PROFILE: {user_profile['level']} | Skills: {skills[:5]}...")
#         print(f">>> SUMMARY USED FOR EMBEDDING: {summary[:120]}...")

#         # ==============================
#         # B2. EMBEDDING + CLUSTERING
#         # ==============================

#         # [THAY ĐỔI QUAN TRỌNG 2]
#         # Thay vì embed toàn bộ clean_text → embed SUMMARY
#         jd_vector = self.embedder.get_embedding(clean_summary)

#         jd_cluster_id = self.cluster_assigner.get_cluster(jd_vector.reshape(1, -1))
#         print(f">>> CLUSTER: Assigned to Group #{jd_cluster_id}")

#         # ==============================
#         # B3. RETRIEVAL (SEARCH)
#         # ==============================

#         # [THAY ĐỔI QUAN TRỌNG 3]
#         # Semantic Search dùng jd_vector (từ summary)
#         # BM25 vẫn dùng clean_text để giữ từ khóa chính xác
#         candidates = self.retriever.get_candidates(
#             query_text=clean_text,
#             query_vector=jd_vector,
#             top_k_each=50
#         )

#         # ==============================
#         # B4. SCORING + RULES
#         # ==============================

#         ranked_results = []

#         for cand in candidates:
#             c_id = cand["id"]
#             if c_id not in self.course_lookup:
#                 continue

#             info = self.course_lookup[c_id]
#             full_cand = {**cand, **info}

#             rule_multiplier = self.rules.apply_rules(user_profile, full_cand)

#             model_score, breakdown = self.scorer.calculate_score(
#                 full_cand,
#                 user_profile,
#                 jd_cluster_id
#             )

#             final_score = model_score * rule_multiplier

#             breakdown["rule_multiplier"] = rule_multiplier
#             breakdown["final_score"] = final_score

#             explanation_json = self.explainer.generate_explanation(
#                 info.get("course_name"),
#                 breakdown
#             )

#             ranked_results.append({
#                 "id": c_id,
#                 "title": info.get("course_name"),
#                 "score": final_score,
#                 "level": info.get("level"),
#                 "url": info.get("page_url"),
#                 "breakdown": breakdown,
#                 "explanation": explanation_json
#             })

#         ranked_results.sort(key=lambda x: x["score"], reverse=True)
#         final_recs = ranked_results[:top_k]

#         return {
#             "time": f"{time.time() - t0:.2f}s",
#             "summary": summary,
#             "profile": user_profile,
#             "recommendations": final_recs
#         }

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--jd", type=str, default="Data Scientist with Python and SQL skills")
#     args = parser.parse_args()
    
#     sys = CourseRecommenderSystem()
#     res = sys.recommend(args.jd)
    
#     print("="*60)
#     print(f"JD SUMMARY: {res['summary']}")
#     print("-" * 60)
#     for i, r in enumerate(res['recommendations']):
#         # Parse JSON giải thích để in ra đẹp hơn
#         expl = json.loads(r['explanation'])
#         reasons = ", ".join(expl['key_factors'])
        
#         print(f"{i+1}. [{r['score']:.4f}] {r['title']} ({r['level']})")
#         print(f"   Lý do: {reasons}")
#         print(f"   Link: {r['url']}\n")


# src/main.py
import os
import sys
import json
import time
import argparse

# Add project src to path if run as script
current_file = os.path.abspath(__file__)
src_dir = os.path.dirname(current_file)
project_root = os.path.dirname(src_dir)
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- IMPORT AN TOÀN ---
def safe_import():
    global JDCleaner, ExperienceMapper, SkillExtractor, JDSummarizer
    global EmbeddingModel, ClusterAssigner
    global SemanticSearcher, BM25Searcher, CandidateUnion
    global Scorer, BusinessRules, Explainability

    try:
        from src.jd_processing.jd_cleaner import JDCleaner
        from src.jd_processing.experience_mapper import ExperienceMapper
        # from src.jd_processing.skill_extractor import SkillExtractor
        # from src.jd_processing.jd_summarizer import JDSummarizer
        from src.embedding.embedding_model import EmbeddingModel
        from src.clustering.cluster_assigner import ClusterAssigner
        from src.retrieval.semantic_search import SemanticSearcher
        from src.retrieval.bm25_retriever import BM25Searcher
        from src.retrieval.candidate_union import CandidateUnion
        from src.recommender.scoring import Scorer
        from src.recommender.rules import BusinessRules
        from src.recommender.explainability import Explainability
        from src.jd_processing.jd_quick_extractor import extract_jd_info

    except ImportError:
        # Fallback for direct module execution
        from jd_processing.jd_cleaner import JDCleaner
        from jd_processing.experience_mapper import ExperienceMapper
        # from jd_processing.skill_extractor import SkillExtractor
        # from jd_processing.jd_summarizer import JDSummarizer
        from embedding.embedding_model import EmbeddingModel
        from clustering.cluster_assigner import ClusterAssigner
        from retrieval.semantic_search import SemanticSearcher
        from retrieval.bm25_retriever import BM25Searcher
        from retrieval.candidate_union import CandidateUnion
        from recommender.scoring import Scorer
        from recommender.rules import BusinessRules
        from recommender.explainability import Explainability

safe_import()

class CourseRecommenderSystem:
    def __init__(self, base_dir=None):
        self.base_dir = base_dir if base_dir else project_root
        print("="*60)
        print(f">>> SYSTEM INITIALIZED (Root: {self.base_dir})")
        print("="*60)

        self.paths = {
            "meta": f"{self.base_dir}/outputs/embeddings/course_meta.csv",
            "faiss": f"{self.base_dir}/outputs/indices/faiss_index.bin",
            "bm25": f"{self.base_dir}/outputs/models/bm25_model.pkl",
            "pca": f"{self.base_dir}/outputs/models/pca_model.pkl",
            "kmeans": f"{self.base_dir}/outputs/models/kmeans_model.pkl",
            "summ_model": f"{self.base_dir}/models",
            "best_weights": f"{self.base_dir}/outputs/models/best_weights.json"
        }

        # 1. Load Metadata
        if os.path.exists(self.paths["meta"]):
            import pandas as pd
            self.course_df = pd.read_csv(self.paths["meta"])
            if 'emb_index' not in self.course_df.columns:
                self.course_df['emb_index'] = self.course_df.index
            # ensure id column exists in lookup (use emb_index as key)
            self.course_lookup = self.course_df.set_index('emb_index').to_dict('index')
            print(f"Metadata: {len(self.course_df)} courses")
        else:
            print("Meta file missing")
            self.course_df = None
            self.course_lookup = {}

        # 2. Load Processors
        print("\n[1/4] Loading Processors...")
        self.cleaner = JDCleaner()
        self.exp_mapper = ExperienceMapper()
        self.skill_extractor = SkillExtractor(self.paths["meta"])
        self.summarizer = JDSummarizer(model_dir=self.paths["summ_model"])

        # 3. Load AI Models
        print("\n[2/4] Loading AI Models...")
        self.embedder = EmbeddingModel()
        self.cluster_assigner = ClusterAssigner(self.paths["pca"], self.paths["kmeans"])

        # 4. Load Search Engines
        print("\n[3/4] Loading Retrieval Engines...")
        self.semantic_searcher = SemanticSearcher(self.paths["faiss"])
        self.bm25_searcher = BM25Searcher(model_path=self.paths["bm25"])
        self.retriever = CandidateUnion(self.semantic_searcher, self.bm25_searcher)

        # 5. Load Logic Modules (Rules & Explainability)
        print("\n[4/4] Loading Logic & Rules...")
        self.rules = BusinessRules()
        self.explainer = Explainability()

        # Load weights tối ưu nếu có (từ weight_search.py)
        weights = None
        if os.path.exists(self.paths["best_weights"]):
            with open(self.paths["best_weights"], 'r') as f:
                weights = json.load(f)
            print("   -> Loaded Optimized Weights from Grid Search!")
        self.scorer = Scorer(weights=weights)

        print("\nREADY!\n")

    # def recommend(self, raw_jd, user_years=None, top_k=10):
    #     """
    #     Main recommend pipeline:
    #      - clean JD
    #      - summarize (LLM with safe fallback)
    #      - extract experience & skills
    #      - cluster assign (embed summary)
    #      - retrieve candidates (semantic + bm25)
    #      - score (model_score) and apply business rules
    #      - return ranked Top-K with breakdown + explanation
    #     """
    #     t0 = time.time()

    #     # B1. Clean JD
    #     clean_text = self.cleaner.clean(raw_jd)

    #     # B2. Summarize with safe wrapper (LLM may fail)
    #     try:
    #         summary = self.summarizer.summarize(clean_text)
    #     except Exception as e:
    #         print(f"[WARN] Summarizer failed: {e}. Falling back to truncated clean text.")
    #         summary = clean_text[:500]

    #     # ensure summary is cleaned for embedding
    #     clean_summary = self.cleaner.clean(summary)

    #     # B3. Experience mapping and skills extraction (use original JD text for extraction)
    #     exp_info = self.exp_mapper.map_experience(clean_text)
    #     skills = self.skill_extractor.extract(clean_text)

    #     # B4. Domain inference (NO HARD-CODE)
    #     try:
    #         domain = self.rules._detect_domain_from_skills(skills)
    #     except Exception:
    #         domain = "general"

    #     # Title: use first line of JD (not hard-coded slice)
    #     first_line = raw_jd.split("\n")[0] if raw_jd else ""
    #     title_excerpt = first_line[:80]

    #     user_profile = {
    #         "level": exp_info.get('level'),
    #         "years_exp": exp_info.get('years'),
    #         "skills": skills,
    #         "domain": domain,
    #         "title": title_excerpt,
    #         "summary": summary,
    #         "user_years": user_years
    #     }

    #     print(f">>> PROFILE: level={user_profile['level']} | domain={domain} | skills={skills[:6]}")
    #     print(f">>> SUMMARY USED FOR EMBEDDING (preview): {summary[:140]}...")

    #     # B5. Embedding + Cluster assign (embed summary)
    #     jd_vector = self.embedder.get_embedding(clean_summary)
    #     jd_cluster_id = self.cluster_assigner.get_cluster(jd_vector.reshape(1, -1))
    #     print(f">>> CLUSTER: Assigned to Group #{jd_cluster_id}")

    #     # B6. Retrieval (semantic + bm25 union)
    #     candidates = self.retriever.get_candidates(query_text=clean_text, query_vector=jd_vector, top_k_each=50)

    #     # B7. Scoring + Business Rules
    #     ranked_results = []
    #     for cand in candidates:
    #         c_id = cand.get("id")
    #         if c_id not in self.course_lookup:
    #             print(f"[WARN] Candidate ID {c_id} not found in course_lookup")
    #             continue

    #         info = self.course_lookup[c_id]
    #         full_cand = {**cand, **info}

    #         # 1) Compute model_score and breakdown from Scorer
    #         model_score, breakdown = self.scorer.calculate_score(full_cand, user_profile, jd_cluster_id)

    #         # 2) Compute rule multiplier (business rules)
    #         rule_multiplier = self.rules.apply_rules(user_profile, full_cand)

    #         # LOG if rules changed score
    #         if rule_multiplier != 1.0:
    #             print(f"[RULE] Applied rule multiplier {rule_multiplier} for course {c_id}")

    #         # 2.5) Add score_before_rules to breakdown for explainability
    #         breakdown["score_before_rules"] = model_score

    #         # 3) Final score after business rules
    #         final_score = model_score * rule_multiplier

    #         # 4) Update breakdown fields consistently
    #         breakdown["model_score"] = round(model_score, 6)
    #         breakdown["rule_multiplier"] = round(rule_multiplier, 6)
    #         breakdown["final_score"] = round(final_score, 6)

    #         # 5) Explainability JSON
    #         explanation_json = self.explainer.generate_explanation(info.get('course_name'), breakdown)

    #         ranked_results.append({
    #             "id": c_id,
    #             "title": info.get('course_name'),
    #             "score": final_score,
    #             "level": info.get('level'),
    #             "url": info.get('page_url'),
    #             "breakdown": breakdown,
    #             "explanation": explanation_json
    #         })

    #     # Sort & take Top-K
    #     ranked_results.sort(key=lambda x: x["score"], reverse=True)
    #     final_recs = ranked_results[:top_k]

    #     return {
    #         "time": f"{time.time() - t0:.2f}s",
    #         "summary": summary,
    #         "profile": user_profile,
    #         "recommendations": final_recs
    #     }

    def recommend(self, raw_jd, user_years=None, top_k=10):
        """
        Pipeline recommend (cập nhật):
        - clean JD
        - gọi extractor 1 lần (LLM -> summary, skills, domain) với fallback
        - extract experience (years/level) bằng ExperienceMapper
        - embed summary, assign cluster
        - retrieve candidates (semantic + bm25)
        - score bằng Scorer, áp dụng BusinessRules
        - trả về Top-K với breakdown + explanation
        """
        t0 = time.time()

        # B1. Clean JD (giữ để dùng cho các bước không dùng LLM)
        clean_text = self.cleaner.clean(raw_jd)

        # B2. Gọi extractor 1 lần: extract_jd_info trả về dict {summary, skills, domain}
        # Nếu OpenAI không cấu hình hoặc extractor lỗi -> hàm sẽ fallback heuristics
        try:
            # import extractor (đảm bảo đường dẫn import tương thích)
            from src.jd_processing.jd_quick_extractor import extract_jd_info
        except Exception:
            # nếu import theo package không được thì thử import đường dẫn tương đối
            try:
                from jd_processing.jd_quick_extractor import extract_jd_info
            except Exception:
                extract_jd_info = None

        summary = ""
        skills = []
        domain = "general"

        if extract_jd_info:
            try:
                # Thử dùng LLM (nếu có API key cài sẵn trong env sẽ dùng tự động)
                extractor_out = extract_jd_info(raw_jd, use_llm=True, openai_api_key=None, llm_model="gpt-3.5-turbo")
                if isinstance(extractor_out, dict):
                    # Chuẩn hoá dữ liệu trả về
                    summary = extractor_out.get("summary") or ""
                    skills = extractor_out.get("skills") or []
                    if isinstance(skills, str):
                        # Nếu LLM trả về chuỗi, tách bằng dấu phẩy/newline
                        skills = [s.strip().lower() for s in re.split(r"[,\n;]+", skills) if s.strip()]
                    else:
                        skills = [str(s).strip().lower() for s in skills if isinstance(s, str) and s.strip()]
                    domain = extractor_out.get("domain") or None
            except Exception as e:
                # Nếu extractor lỗi, in warn và fallback xuống phương pháp cũ
                print(f"[WARN] extract_jd_info failed: {e}")
                extractor_out = None

        # Nếu LLM/extractor không trả được dữ liệu hợp lệ -> fallback về những component hiện có
        if not summary:
            try:
                summary = self.summarizer.summarize(clean_text)
            except Exception:
                summary = clean_text[:500]

        if not skills:
            try:
                skills = self.skill_extractor.extract(clean_text)
                # ensure normalized list
                if isinstance(skills, str):
                    skills = [s.strip().lower() for s in re.split(r"[,\n;]+", skills) if s.strip()]
                else:
                    skills = [str(s).strip().lower() for s in skills if isinstance(s, str) and s.strip()]
            except Exception:
                # fallback lightweight extractor if needed
                try:
                    from src.jd_processing.jd_quick_extractor import _simple_skill_extractor
                    skills = _simple_skill_extractor(clean_text, top_n=12)
                except Exception:
                    skills = []

        # Nếu domain chưa có từ extractor, dùng rules._detect_domain_from_skills
        if not domain:
            try:
                domain = self.rules._detect_domain_from_skills(skills)
            except Exception:
                domain = "general"

        # B3. Experience mapping (dùng exp_mapper để detect years/level từ clean_text)
        try:
            exp_info = self.exp_mapper.map_experience(clean_text)
        except Exception as e:
            print(f"[WARN] ExperienceMapper failed: {e}")
            exp_info = {"years": None, "level": None}

        # Title: dùng dòng đầu tiên của raw_jd làm title_excerpt (không cắt nội dung khác)
        first_line = raw_jd.split("\n")[0] if raw_jd else ""
        title_excerpt = first_line[:80]

        # Build user_profile như cũ nhưng dùng kết quả từ extractor
        user_profile = {
            "level": exp_info.get("level"),
            "years_exp": exp_info.get("years"),
            "skills": skills,
            "domain": domain,
            "title": title_excerpt,
            "summary": summary,
            "user_years": user_years
        }

        print(f">>> PROFILE: level={user_profile['level']} | domain={domain} | skills={skills[:6]}")
        print(f">>> SUMMARY USED FOR EMBEDDING (preview): {summary[:140]}...")

        # B4. Chuẩn bị embedding: embed summary (cleaned)
        clean_summary = self.cleaner.clean(summary)
        jd_vector = self.embedder.get_embedding(clean_summary)
        jd_cluster_id = self.cluster_assigner.get_cluster(jd_vector.reshape(1, -1))
        print(f">>> CLUSTER: Assigned to Group #{jd_cluster_id}")

        # B5. Retrieval (semantic + bm25 union)
        candidates = self.retriever.get_candidates(query_text=clean_text, query_vector=jd_vector, top_k_each=50)

        # B6. Scoring + Business Rules
        ranked_results = []
        for cand in candidates:
            c_id = cand.get("id")
            if c_id not in self.course_lookup:
                print(f"[WARN] Candidate ID {c_id} not found in course_lookup")
                continue

            info = self.course_lookup[c_id]
            full_cand = {**cand, **info}

            # 1) Compute model_score and breakdown from Scorer
            model_score, breakdown = self.scorer.calculate_score(full_cand, user_profile, jd_cluster_id)

            # 2) Compute rule multiplier (business rules)
            rule_multiplier = self.rules.apply_rules(user_profile, full_cand)

            # LOG nếu rules thay đổi score
            if rule_multiplier != 1.0:
                print(f"[RULE] Applied rule multiplier {rule_multiplier} for course {c_id}")

            # 2.5) Add score_before_rules to breakdown cho explainability
            breakdown["score_before_rules"] = model_score

            # 3) Final score after business rules
            final_score = model_score * rule_multiplier

            # 4) Update breakdown fields consistent
            breakdown["model_score"] = round(model_score, 6)
            breakdown["rule_multiplier"] = round(rule_multiplier, 6)
            breakdown["final_score"] = round(final_score, 6)

            # 5) Explainability JSON
            explanation_json = self.explainer.generate_explanation(info.get('course_name'), breakdown)

            ranked_results.append({
                "id": c_id,
                "title": info.get('course_name'),
                "score": final_score,
                "level": info.get('level'),
                "url": info.get('page_url'),
                "breakdown": breakdown,
                "explanation": explanation_json
            })

        # Sort & take Top-K
        ranked_results.sort(key=lambda x: x["score"], reverse=True)
        final_recs = ranked_results[:top_k]

        return {
            "time": f"{time.time() - t0:.2f}s",
            "summary": summary,
            "profile": user_profile,
            "recommendations": final_recs
        }



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--jd", type=str, default="Data Scientist with Python and SQL skills")
    parser.add_argument("--top_k", type=int, default=10)
    args = parser.parse_args()

    sys = CourseRecommenderSystem()
    res = sys.recommend(args.jd, top_k=args.top_k)

    print("=" * 60)
    print(f"JD SUMMARY: {res['summary']}")
    print("-" * 60)
    for i, r in enumerate(res['recommendations']):
        # Parse JSON giải thích để in ra đẹp hơn
        try:
            expl = json.loads(r['explanation'])
            reasons = ", ".join(expl.get('key_factors', []))
        except Exception:
            reasons = ""
        print(f"{i+1}. [{r['score']:.4f}] {r['title']} ({r.get('level','N/A')})")
        if reasons:
            print(f"   Lý do: {reasons}")
        print(f"   Link: {r.get('url')}\n")
