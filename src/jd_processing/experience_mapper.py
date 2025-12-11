# import re
# import math

# class ExperienceMapper:
#     def __init__(self):
#         # Định nghĩa các ngưỡng kinh nghiệm
#         self.LEVEL_THRESHOLDS = {
#             'Beginner': 0,
#             'Intermediate': 2,
#             'Advanced': 5,
#             'Expert': 10
#         }

#     def map_experience(self, clean_jd_text):
#         text = clean_jd_text.lower()
#         max_years = 0
#         min_years = 0
        
#         # 1. Tìm phạm vi (X-Y years) hoặc số đơn lẻ (X+ years)
        
#         # Regex mở rộng để bắt các định dạng: "1-3 years", "2-5+ years", "5+ years", "3 years"
#         # Bắt các cặp số, ví dụ: 1-3, 2-5, hoặc số đơn 3, 5
#         matches = re.findall(r'(\d+)\s*[-–]?\s*(\d+)\+?\s*year|(\d+)\+?\s*year', text)
        
#         years_found = []
        
#         for match in matches:
#             # Match có thể là (min, max, None) hoặc (None, None, single)
#             if match[0] and match[1]: # Dạng X-Y
#                 years_found.append(int(match[0]))
#                 years_found.append(int(match[1]))
#             elif match[2]: # Dạng X+ hoặc X
#                 years_found.append(int(match[2]))

#         if years_found:
#             max_years = max(years_found)
#             min_years = min(years_found)
        
#         # 2. Phân loại Level dựa trên max_years (mục tiêu)
#         level = 'Beginner'
#         if max_years >= self.LEVEL_THRESHOLDS['Expert']:
#             level = 'Expert'
#         elif max_years >= self.LEVEL_THRESHOLDS['Advanced']:
#             level = 'Advanced'
#         elif max_years >= self.LEVEL_THRESHOLDS['Intermediate']:
#             level = 'Intermediate'
            
#         # 3. Xử lý các từ khóa nếu không tìm thấy số năm (0 năm)
#         if max_years == 0:
#             if 'senior' in text or 'lead' in text:
#                 level = 'Advanced'
#             elif 'junior' in text or 'fresher' in text:
#                 level = 'Beginner'
#             elif 'entry-level' in text:
#                 level = 'Beginner'

#         # Trả về số năm cao nhất (3 năm) và Level tương ứng (Intermediate)
#         return {
#             "years": max_years,
#             "level": level
#         }

import re

class ExperienceMapper:
    def __init__(self):
        self.LEVEL_THRESHOLDS = {
            'Beginner': 0,
            'Intermediate': 2,
            'Advanced': 5,
            'Expert': 10
        }

        # Regex mạnh hơn – bắt hầu hết JD thực tế
        self.year_pattern = re.compile(
            r"""
            (?P<min>\d+)\s*[-–]\s*(?P<max>\d+)\s*(?:\+)?\s*(?:year|years|yrs|yoe)     # dạng 2-5 years
            |
            (?P<single>\d+)\s*\+\s*(?:year|years|yrs|yoe)                               # dạng 3+ years
            |
            (?P<plain>\d+)\s*(?:year|years|yrs|yoe)                                     # dạng 3 years
            """,
            re.IGNORECASE | re.VERBOSE,
        )

    def map_experience(self, clean_jd_text):
        text = clean_jd_text.lower()

        years_found = []

        # 1. Tìm tất cả match và gom năm lại
        for match in self.year_pattern.finditer(text):
            if match.group("min") and match.group("max"):
                years_found.append(int(match.group("min")))
                years_found.append(int(match.group("max")))
            elif match.group("single"):
                years_found.append(int(match.group("single")))
            elif match.group("plain"):
                years_found.append(int(match.group("plain")))

        max_years = max(years_found) if years_found else 0

        # 2. Suy level từ số năm
        if max_years >= self.LEVEL_THRESHOLDS['Expert']:
            level = 'Expert'
        elif max_years >= self.LEVEL_THRESHOLDS['Advanced']:
            level = 'Advanced'
        elif max_years >= self.LEVEL_THRESHOLDS['Intermediate']:
            level = 'Intermediate'
        else:
            level = 'Beginner'

        # 3. Nếu không có số -> dùng từ khóa
        if max_years == 0:
            if "senior" in text or "lead" in text:
                level = 'Advanced'
            elif "mid" in text or "intermediate" in text:
                level = 'Intermediate'
            elif "junior" in text or "entry" in text or "fresher" in text:
                level = 'Beginner'

        return {
            "years": max_years,
            "level": level
        }
