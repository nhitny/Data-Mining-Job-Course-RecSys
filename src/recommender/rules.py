# src/recommender/rules.py
"""
BusinessRules (phiên bản LLM-first)

Mục tiêu:
- Dùng trực tiếp skills do LLM trích xuất (user_profile['skills']) làm nguồn truth.
- Chỉ thực hiện chuẩn hoá nhẹ (lower, strip, loại punctuation thừa) và dedupe,
  tránh thay đổi ý nghĩa do aliasing nặng.
- Nếu candidate có trường skills (list/string) thì ưu tiên dùng nó. Nếu không,
  fallback sang tokenization từ course_name/clean_text.
- Ghi matched_skills và exp_multiplier vào candidate_item để debug/explain.
"""

import re

# -------------------------------------------------------------------
# Hàm phụ: compute_experience_multiplier (giữ nguyên logic bạn đang dùng)
# -------------------------------------------------------------------
def compute_experience_multiplier(user_years, jd_years):
    # Nếu không có thông tin thì không ảnh hưởng
    if user_years is None or jd_years is None:
        return 1.0

    try:
        u = float(user_years)
        j = float(jd_years)
    except Exception:
        # Nếu parse lỗi, không thay đổi factor
        return 1.0

    diff = u - j

    # User đáp ứng hoặc vượt yêu cầu -> boost nhẹ
    if diff >= 0:
        return 1.05

    # Thấp hơn nhưng ít hơn 1 năm -> neutral (không phạt)
    if diff > -1:
        return 1.0

    # Thấp hơn 1-3 năm -> phạt nhẹ
    if diff > -3:
        return 0.9

    # Thấp hơn >3 năm -> phạt mạnh
    return 0.7


# -------------------------------------------------------------------
# BusinessRules: LLM-first
# -------------------------------------------------------------------
class BusinessRules:
    def __init__(self):
        # cấu hình multiplier mặc định; có thể load từ file config nếu muốn
        self.multipliers = {
            # Nếu khóa học không chứa bất kỳ core skill nào -> phạt mạnh
            "missing_core_penalty": 0.5,
            # Nếu JD yêu cầu advanced mà khóa học là beginner -> phạt
            "jd_adv_course_beginner_penalty": 0.7,
            # Nếu JD là beginner mà khóa học advanced -> phạt
            "jd_beginner_course_adv_penalty": 0.5,
            # Nếu user muốn certificate và khóa có certificate -> boost
            "certificate_boost": 1.2
        }

    # -------------- helpers --------------
    def _simple_normalize(self, s):
        """
        Chuẩn hoá nhẹ một skill/chuỗi:
        - lowercase, strip
        - loại bớt ký tự punctuation thừa (nhưng giữ multi-word)
        """
        if not s:
            return ""
        t = str(s).strip().lower()
        # thay các dấu ngoặc, dấu câu thành space (giữ từ nhiều token)
        t = re.sub(r"[()\[\]\{\}\.,;:/\\\*\"']", " ", t)
        # collapse whitespace
        t = re.sub(r"\s+", " ", t).strip()
        return t

    def _normalize_skills_from_llm(self, raw_skills):
        """
        Nhận giá trị skills do LLM trả (có thể là list hoặc string).
        Trả về list đã chuẩn hoá theo thứ tự xuất hiện (dedupe).
        Lưu ý: không ép map alias nặng — để giữ đúng output của LLM.
        """
        out = []
        if raw_skills is None:
            return out
        if isinstance(raw_skills, str):
            # tách nếu LLM trả chuỗi comma separated hoặc newline
            candidates = [s.strip() for s in re.split(r"[,\n;]+", raw_skills) if s.strip()]
        elif isinstance(raw_skills, (list, tuple)):
            candidates = list(raw_skills)
        else:
            # cố chuyển thành string rồi split
            candidates = [str(raw_skills)]

        for s in candidates:
            n = self._simple_normalize(s)
            if n and n not in out:
                out.append(n)
        return out

    def _normalize_candidate_skills(self, candidate_item):
        """
        Lấy thông tin skills từ candidate_item nếu có (ưu tiên các trường chuyên biệt).
        Trả về list chuẩn hoá (có thể trống).
        """
        for k in ("skills", "course_skills", "skills_list", "skills_gain_clean"):
            v = candidate_item.get(k)
            if not v:
                continue
            # nếu list/tuple -> chuẩn hoá từng phần tử
            if isinstance(v, (list, tuple)):
                out = []
                for s in v:
                    n = self._simple_normalize(s)
                    if n and n not in out:
                        out.append(n)
                return out
            # nếu string -> split
            if isinstance(v, str):
                parts = [s.strip() for s in re.split(r"[,\n;]+", v) if s.strip()]
                out = []
                for s in parts:
                    n = self._simple_normalize(s)
                    if n and n not in out:
                        out.append(n)
                return out
            # fallback: stringify
            try:
                txt = str(v)
                parts = [s.strip() for s in re.split(r"[,\n;]+", txt) if s.strip()]
                out = []
                for s in parts:
                    n = self._simple_normalize(s)
                    if n and n not in out:
                        out.append(n)
                return out
            except Exception:
                continue
        # không có field skills explicit -> trả rỗng để caller fallback
        return []

    def _tokenize_course_text(self, candidate_item):
        """
        Fallback: tách token từ course_name + clean_text + các trường skills nếu có (stringified),
        trả về danh sách token đã chuẩn hoá, unique.
        """
        txt = (str(candidate_item.get('course_name', '')) + " " + str(candidate_item.get('clean_text', ''))).lower()
        for k in ("skills", "course_skills", "skills_list", "skills_gain_clean"):
            if candidate_item.get(k):
                txt += " " + str(candidate_item.get(k))
        # split theo non-alnum (giữ +,#)
        tokens = [t for t in re.split(r"[^a-z0-9\+\#]+", txt) if t]
        out = []
        for t in tokens:
            n = self._simple_normalize(t)
            if n and n not in out:
                out.append(n)
        return out

    # -------------- domain heuristic (fallback) --------------
    def _detect_domain_from_skills(self, jd_skills):
        """
        Heuristic nhẹ nếu domain không có trong user_profile.
        jd_skills là list đã chuẩn hoá.
        """
        if not jd_skills:
            return "general"
        jd = set(s.lower() for s in jd_skills)
        DOMAIN_KEYWORDS = {
            "robotics": {"ros", "ros2", "slam", "opencv", "pytorch", "tensorflow", "navigation", "robot"},
            "data": {"machine learning", "deep learning", "sql", "pandas", "python", "nlp", "transformers"},
            "ba": {"business analyst", "bpmn", "requirements", "elicitation", "stakeholder", "agile", "scrum"},
            "devops": {"docker", "kubernetes", "ci", "cd", "jenkins", "terraform", "ansible", "mlops"},
            "software": {"java", "c++", "spring", "react", "api", "backend", "microservices"},
            "cloud": {"aws", "azure", "gcp", "cloud", "lambda"}
        }
        scores = {d: len(jd.intersection(kws)) for d, kws in DOMAIN_KEYWORDS.items()}
        best = max(scores, key=scores.get)
        if scores[best] == 0:
            return "general"
        return best

    # -------------- main apply_rules --------------
    def apply_rules(self, user_profile, candidate_item):
        """
        Áp dụng các rule lên một ứng viên (candidate_item) dựa trên user_profile.
        - user_profile['skills'] được coi là nguồn truth (được LLM trích xuất).
        - candidate_item có thể có trường skills; nếu không có thì fallback tokenization.
        - Trả về float factor để nhân vào điểm.
        """
        factor = 1.0

        # Lấy skills từ LLM (ưu tiên) và chuẩn hoá nhẹ
        raw_jd_skills = user_profile.get('skills') or []
        jd_skills = self._normalize_skills_from_llm(raw_jd_skills)

        # domain: ưu tiên domain do LLM trả nếu có, ngược lại heuristic
        jd_domain = user_profile.get('domain') or ""
        if not jd_domain or not isinstance(jd_domain, str) or jd_domain.strip() == "":
            jd_domain = self._detect_domain_from_skills(jd_skills)
        jd_domain = str(jd_domain).lower()

        # level
        jd_level_raw = user_profile.get('level') or ""
        jd_level = str(jd_level_raw).lower()
        course_level = str(candidate_item.get('level') or "").lower()

        # Lấy hoặc token hóa skill của candidate
        course_skills = self._normalize_candidate_skills(candidate_item)
        if not course_skills:
            course_skills = self._tokenize_course_text(candidate_item)

        # so sánh bằng set intersection
        jd_core_set = set(jd_skills)
        course_set = set(course_skills)

        matched = list(sorted(jd_core_set.intersection(course_set)))
        # lưu matched skills vào candidate để explainability
        try:
            candidate_item["matched_skills"] = matched
        except Exception:
            pass

        # Nếu JD có core skills nhưng khóa học không chứa bất kỳ core skill nào -> phạt nặng
        if jd_core_set:
            if len(matched) == 0:
                factor *= float(self.multipliers.get("missing_core_penalty", 0.5))
            else:
                # nếu chỉ match rất ít (ví dụ <20%) -> phạt nhẹ
                ratio = len(matched) / max(1.0, len(jd_core_set))
                if ratio < 0.2:
                    factor *= 0.9

        # Level mismatch penalties (giữ logic cũ)
        if "advanced" in jd_level and "beginner" in course_level:
            factor *= float(self.multipliers.get("jd_adv_course_beginner_penalty", 0.7))
        if "beginner" in jd_level and "advanced" in course_level:
            factor *= float(self.multipliers.get("jd_beginner_course_adv_penalty", 0.5))

        # Certificate preference
        user_title = str(user_profile.get('title', '')).lower()
        user_summary = str(user_profile.get('summary', '')).lower()
        if 'certificate' in user_title or 'certificate' in user_summary or user_profile.get('prefers_certificate'):
            course_text = (str(candidate_item.get('course_name', '')) + " " + str(candidate_item.get('clean_text', ''))).lower()
            if 'certificate' in course_text or 'certification' in course_text:
                factor *= float(self.multipliers.get("certificate_boost", 1.2))

        # Experience multiplier (áp dụng cuối cùng)
        user_years = user_profile.get("user_years")
        jd_years = user_profile.get("years_exp")
        exp_mult = compute_experience_multiplier(user_years, jd_years)
        factor *= exp_mult
        try:
            candidate_item["exp_multiplier"] = float(exp_mult)
        except Exception:
            candidate_item["exp_multiplier"] = exp_mult

        # Trả về factor cuối cùng
        return float(factor)
