from recommendation_path import CourseRecommender
import pandas as pd
import numpy as np

def calculate_score(path_result, expected_keywords, user_type):
    """
    Hàm chấm điểm chất lượng của 1 lộ trình gợi ý.
    """
    score = 0
    total_courses = 0
    
    for stage, items in path_result.items():
        for item in items:
            total_courses += 1
            course_name = str(item['data']['course_name']).lower()
            level = str(item['data'].get('level_std', '')).lower()
            
            # 1. Tiêu chí Từ khóa (Quan trọng nhất)
            for kw in expected_keywords:
                if kw in course_name: 
                    score += 1.0 # Cộng 1 điểm nếu đúng từ khóa
                    break
            
            # 2. Tiêu chí Đúng trình độ (Coherence)
            if user_type == 'senior':
                if 'advanced' in level: score += 0.5
                if 'beginner' in level: score -= 1.0 # Phạt nặng nếu Senior mà ra Beginner
            elif user_type == 'fresher':
                if 'beginner' in level: score += 0.5
                if 'advanced' in level: score -= 0.5

    # Chuẩn hóa điểm về thang 0-1
    return max(0, score / total_courses) if total_courses > 0 else 0

def run_grid_search():
    print(" BẮT ĐẦU TỐI ƯU HÓA (GRID SEARCH)...")
    recsys = CourseRecommender()
    
    # 1. Định nghĩa tập dữ liệu mẫu (Golden Set)
    test_cases = [
        {
            'jd': "AI Engineer Python Deep Learning Biology",
            'user': {'years_experience': 8, 'known_skills': ['python']}, # Senior
            'keywords': ['biology', 'genomics', 'deep learning'],
            'type': 'senior'
        },
        {
            'jd': "Digital Marketing SEO Content",
            'user': {'years_experience': 0, 'known_skills': []}, # Fresher
            'keywords': ['marketing', 'seo'],
            'type': 'fresher'
        }
    ]
    
    # 2. Định nghĩa không gian tìm kiếm (Grid)
    # Thử các mức độ phạt/thưởng khác nhau
    param_grid = [
        # Bộ 1: Nhẹ nhàng (Baseline)
        {'senior_penalty': 0.8, 'senior_boost': 1.1, 'fresher_boost': 1.1, 'skill_gap_penalty': 0.9, 'keyword_boost': 1.2},
        
        # Bộ 2: Vừa phải (Hiện tại)
        {'senior_penalty': 0.6, 'senior_boost': 1.3, 'fresher_boost': 1.2, 'skill_gap_penalty': 0.8, 'keyword_boost': 1.4},
        
        # Bộ 3: Cực đoan (Phạt nặng, thưởng đậm)
        {'senior_penalty': 0.3, 'senior_boost': 1.5, 'fresher_boost': 1.4, 'skill_gap_penalty': 0.5, 'keyword_boost': 1.8},
    ]
    
    best_score = -1
    best_params = {}
    
    # 3. Chạy vòng lặp
    for i, params in enumerate(param_grid):
        print(f"\n Thử nghiệm Bộ tham số #{i+1}: {params}")
        
        current_total_score = 0
        
        for case in test_cases:
            # Chạy gợi ý với params hiện tại
            path = recsys.recommend(
                case['jd'], 
                case['user'], 
                top_k=5, 
                boost_keywords=case['keywords'],
                weights=params # <--- Truyền weights vào đây
            )
            
            # Chấm điểm kết quả
            s = calculate_score(path, case['keywords'], case['type'])
            current_total_score += s
            
        avg_score = current_total_score / len(test_cases)
        print(f"   -> Điểm trung bình: {avg_score:.4f}")
        
        if avg_score > best_score:
            best_score = avg_score
            best_params = params
            
    # 4. Kết luận
    print("\n" + "="*60)
    print(f" KẾT QUẢ TỐI ƯU NHẤT")
    print(f"   Điểm số: {best_score:.4f}")
    print(f"   Bộ tham số: {best_params}")
    print("="*60)
    
    return best_params

if __name__ == "__main__":
    run_grid_search()