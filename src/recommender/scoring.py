# # import numpy as np

# # class Scorer:
# #     def __init__(self, weights=None):
# #         # Cấu hình trọng số mặc định
# #         self.weights = weights or {
# #             "semantic": 0.7,        
# #             "bm25": 0.3,            
# #             "cluster_boost": 1.15,  
# #             "level_boost": 1.1,     
# #             "skill_bonus": 0.05     
# #         }

# #     def calculate_score(self, candidate, user_profile, jd_cluster_id):
# #         """
# #         Tính điểm cuối cùng và trả về breakdown chi tiết
# #         """
# #         # 1. Base Score
# #         semantic_score = candidate.get('semantic_score', 0)
# #         bm25_score = candidate.get('bm25_score', 0)
        
# #         # Công thức: semantic * weight + bm25 * weight (normalize nếu cần)
# #         # Ở đây ta lấy semantic làm gốc
# #         base_score = semantic_score 
        
# #         final_score = base_score
        
# #         # 2. Cluster Boost
# #         cluster_match = False
# #         if 'cluster_id' in candidate:
# #             if candidate['cluster_id'] == jd_cluster_id:
# #                 final_score *= self.weights.get('cluster_boost', 1.0)
# #                 cluster_match = True
        
# #         # 3. Level Boost
# #         level_match = False
# #         course_level = str(candidate.get('level', '')).lower()
# #         if str(user_profile['level']).lower() in course_level:
# #             final_score *= self.weights.get('level_boost', 1.0)
# #             level_match = True
            
# #         # 4. Skill Boost
# #         course_text = str(candidate.get('clean_text', '')).lower()
# #         matched_skills = [s for s in user_profile['skills'] if s in course_text]
        
# #         skill_multiplier = 1 + (len(matched_skills) * self.weights.get('skill_bonus', 0.05))
# #         final_score *= skill_multiplier
        
# #         # --- QUAN TRỌNG: Trả về đầy đủ các trường mà Explainability cần ---
# #         explanation = {
# #             "final_score": final_score,
# #             "base_score": round(base_score, 4),
            
# #             # Các trường bắt buộc cho explainability.py:
# #             "semantic_contrib": semantic_score, # Giả lập contribution
# #             "skill_contrib": len(matched_skills) * 0.05,
# #             "rule_multiplier": skill_multiplier, # Hoặc tổng hợp các multiplier
            
# #             # Các trường thông tin thêm
# #             "cluster_match": cluster_match,
# #             "level_match": level_match,
# #             "matched_skills": matched_skills,
# #             "skill_boost_factor": round(skill_multiplier, 2)
# #         }
        
# #         return final_score, explanation


# # scoring.py
# import numpy as np
# import re

# class Scorer:
#     def __init__(self, weights=None):
#         # Giữ gần nguyên weights gốc, nhưng sẽ kết hợp bm25 đúng
#         self.weights = weights or {
#             "semantic": 0.7,
#             "bm25": 0.3,
#             "cluster_boost": 1.15,
#             "level_boost": 1.1,
#             "skill_bonus": 0.05
#         }
#         # cap tránh runaway
#         self.max_skill_multiplier = 1.5

#     def _normalize_level(self, text):
#         if not text: return ""
#         t = str(text).lower()
#         if "senior" in t or t.startswith("sr") or "expert" in t:
#             return "senior"
#         if "mid" in t or "intermed" in t:
#             return "mid"
#         if "junior" in t or "entry" in t or "beginner" in t:
#             return "junior"
#         if "all" in t:
#             return "all"
#         return t

#     def _get_course_skill_set(self, candidate):
#         # ưu tiên explicit skills fields
#         for k in ("skills", "skills_gain_clean", "skills_list", "course_skills"):
#             if candidate.get(k):
#                 txt = str(candidate[k])
#                 return set([t.strip().lower() for t in re.findall(r"[\w\+\#\.]+", txt) if len(t.strip())>1])
#         # fallback to text
#         txt = str(candidate.get("clean_text","")) + " " + str(candidate.get("course_name",""))
#         return set([t.strip().lower() for t in re.findall(r"[\w\+\#\.]+", txt) if len(t.strip())>1])

#     def calculate_score(self, candidate, user_profile, jd_cluster_id):
#         """
#         Tính điểm và trả về (final_score, breakdown)
#         Breakdown luôn chứa 'skill_contrib' để tránh KeyError ở caller.
#         """
#         # 1. Base combo: combine semantic + bm25 using provided weights
#         semantic_score = float(candidate.get('semantic_score', 0) or 0)
#         bm25_score = float(candidate.get('bm25_score', 0) or 0)

#         w_sem = float(self.weights.get("semantic", 0.7))
#         w_bm = float(self.weights.get("bm25", 0.3))
#         total = w_sem + w_bm if (w_sem + w_bm) > 0 else 1.0
#         w_sem /= total
#         w_bm /= total

#         base_score = semantic_score * w_sem + bm25_score * w_bm
#         final_score = base_score

#         # 2. Cluster boost (multiplicative)
#         cluster_match = False
#         if candidate.get('cluster_id') is not None and jd_cluster_id is not None:
#             if candidate.get('cluster_id') == jd_cluster_id:
#                 final_score *= self.weights.get('cluster_boost', 1.0)
#                 cluster_match = True

#         # 3. Level matching (normalize levels)
#         level_match = False
#         user_lvl = self._normalize_level(user_profile.get('level', ''))
#         course_lvl = self._normalize_level(candidate.get('level', ''))

#         if user_lvl and course_lvl and user_lvl == course_lvl:
#             final_score *= self.weights.get('level_boost', 1.0)
#             level_match = True
#         else:
#             # nhẹ phạt mismatch extreme
#             if user_lvl == 'senior' and course_lvl == 'junior':
#                 final_score *= 0.85
#             if user_lvl == 'junior' and course_lvl == 'senior':
#                 final_score *= 0.9

#         # 4. Skill matching — prefer explicit skill field
#         course_skill_set = self._get_course_skill_set(candidate)
#         jd_skills = [s.lower() for s in (user_profile.get('skills') or [])]

#         overlap = set(jd_skills).intersection(course_skill_set)
#         skill_count = len(overlap)
#         skill_ratio = skill_count / max(1, len(jd_skills))

#         # skill_effect: additive (so không phóng đại quá mức)
#         skill_effect = skill_ratio * self.weights.get('skill_bonus', 0.05) * base_score
#         final_score += skill_effect

#         # 5. Cap uplift to avoid runaway
#         if base_score > 0:
#             uplift = final_score / base_score
#             if uplift > self.max_skill_multiplier:
#                 final_score = base_score * self.max_skill_multiplier

#         # 6. Breakdown (include skill_contrib key)
#         if base_score > 0:
#             skill_contrib = skill_effect / base_score
#         else:
#             skill_contrib = float(skill_effect)  # fallback absolute

#         breakdown = {
#             "final_score": round(float(final_score), 6),
#             "base_score": round(float(base_score), 6),
#             "semantic_contrib": float(semantic_score),
#             "bm25_contrib": float(bm25_score),
#             "skill_effect": round(float(skill_effect), 6),
#             "skill_contrib": round(float(skill_contrib), 6),   # <-- required key
#             "skill_count": int(skill_count),
#             "skill_ratio": round(float(skill_ratio), 4),
#             "matched_skills": list(overlap),
#             "cluster_match": bool(cluster_match),
#             "level_match": bool(level_match),
#             "skill_boost_factor": round(float(final_score / base_score) if base_score>0 else 1.0, 3)
#         }

#         return float(final_score), breakdown

import numpy as np
import re

class Scorer:
    """
    Scorer kết hợp:
    - semantic_score
    - bm25_score
    - cluster alignment
    - level alignment
    - skill matching

    Trả về:
    - final_score
    - breakdown (dùng cho Explainability + UI)
    """

    def __init__(self, weights=None):

        # Trọng số mặc định (có thể override bằng best_weights.json)
        self.weights = weights or {
            "semantic": 0.6,
            "bm25": 0.4,
            "cluster_boost": 1.10,
            "level_boost": 1.05,
            "skill_bonus": 0.05
        }

    # -----------------------------
    # Helper: chuẩn hóa level
    # -----------------------------
    def _normalize_level(self, text):
        if not text: return ""
        t = str(text).lower()
        if "senior" in t or "expert" in t or t.startswith("sr"):
            return "advanced"
        if "mid" in t or "inter" in t:
            return "intermediate"
        if "junior" in t or "entry" in t or "beginner" in t:
            return "beginner"
        return t

    # -----------------------------
    # Helper: lấy skill từ course
    # -----------------------------
    def _get_course_skill_set(self, candidate):
        # ưu tiên field structured
        for k in ("skills", "skills_list", "skills_gain_clean", "course_skills"):
            if candidate.get(k):
                txt = str(candidate[k]).lower()
                return set(re.findall(r"[a-zA-Z0-9\+\#\.]+", txt))

        # fallback: clean_text
        txt = (
            str(candidate.get("clean_text", "")).lower() + " " +
            str(candidate.get("course_name", "")).lower()
        )
        return set(re.findall(r"[a-zA-Z0-9\+\#\.]+", txt))

    # -----------------------------
    # MAIN FUNCTION
    # -----------------------------
    def calculate_score(self, candidate, user_profile, jd_cluster_id):
        """
        Return:
            model_score  (float)
            breakdown    (dict)
        """

        # ============================================
        # 1. SEMANTIC + BM25 COMBINATION
        # ============================================

        semantic = float(candidate.get("semantic_score", 0))
        bm25 = float(candidate.get("bm25_score", 0))

        w_sem = self.weights.get("semantic", 0.6)
        w_bm = self.weights.get("bm25", 0.4)
        norm = w_sem + w_bm

        w_sem /= norm
        w_bm /= norm

        model_score = semantic * w_sem + bm25 * w_bm

        # ============================================
        # 2. CLUSTER BOOST (multiplicative)
        # ============================================

        cluster_match = False
        if candidate.get("cluster_id") == jd_cluster_id:
            model_score *= self.weights.get("cluster_boost", 1.0)
            cluster_match = True

        # ============================================
        # 3. LEVEL BOOST
        # ============================================

        level_match = False
        user_lvl = self._normalize_level(user_profile.get("level"))
        course_lvl = self._normalize_level(candidate.get("level"))

        if user_lvl and course_lvl and user_lvl == course_lvl:
            model_score *= self.weights.get("level_boost", 1.0)
            level_match = True

        # ============================================
        # 4. SKILL MATCHING (ADD-ON, không nhân)
        # ============================================

        course_skill_set = self._get_course_skill_set(candidate)
        jd_skills = [s.lower() for s in (user_profile.get("skills") or [])]

        overlap = set(jd_skills).intersection(course_skill_set)
        skill_ratio = len(overlap) / max(1, len(jd_skills))

        skill_effect = skill_ratio * self.weights.get("skill_bonus", 0.05)

        # additive (ổn định hơn multiplicative)
        model_score += skill_effect

        # ============================================
        # 5. BREAKDOWN
        # ============================================

        breakdown = {
            "model_score": round(float(model_score), 6),
            "semantic_contrib": semantic,
            "bm25_contrib": bm25,

            "cluster_match": cluster_match,
            "level_match": level_match,

            "skill_contrib": round(float(skill_effect), 6),
            "matched_skills": list(overlap),
            "skill_ratio": round(float(skill_ratio), 4),
        }

        return float(model_score), breakdown

