import streamlit as st
import time
import sys
import os

# --- 1. SETUP ĐƯỜNG DẪN IMPORT ---
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from recommendation_path import CourseRecommender
except ImportError:
    st.error("Không tìm thấy file 'recommendation_path.py'.")
    st.stop()

# --- 2. HÀM TRÍCH XUẤT TỪ KHÓA ---
from sklearn.feature_extraction.text import TfidfVectorizer
def extract_keywords(text, top_n=5):
    try:
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform([text])
        feature_names = vectorizer.get_feature_names_out()
        dense = tfidf_matrix.todense()
        episode = dense[0].tolist()[0]
        phrase_scores = [pair for pair in zip(range(0, len(episode)), episode) if pair[1] > 0]
        sorted_phrase_scores = sorted(phrase_scores, key=lambda t: t[1] * -1)
        keywords = []
        for phrase, score in sorted_phrase_scores[:top_n]:
            keywords.append(feature_names[phrase])
        return keywords
    except:
        return []

# --- 3. CẤU HÌNH TRANG ---
st.set_page_config(page_title="Course.AI", page_icon="🎓", layout="wide")

st.markdown("""
<style>
    .stage-header { 
        background: linear-gradient(90deg, #2c3e50 0%, #4ca1af 100%); 
        padding: 10px 20px; 
        border-radius: 8px; 
        margin: 25px 0 15px 0; 
        color: white; 
        font-weight: bold;
        font-size: 1.1em;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .course-card { 
        background: white; 
        border: 1px solid #ddd; 
        border-radius: 12px; 
        padding: 20px; 
        height: 100%; 
        box-shadow: 0 4px 6px rgba(0,0,0,0.05); 
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    .course-card:hover { 
        border-color: #4ca1af; 
        transform: translateY(-5px);
        box-shadow: 0 10px 20px rgba(0,0,0,0.1);
    }
    .course-title {
        color: #111827 !important; 
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: 800;
        font-size: 1.2rem; /* Tăng size chữ */
        margin-bottom: 12px;
        line-height: 1.3;
        text-transform: capitalize;
    }
    .badge-container {
        display: flex;
        gap: 10px;
        margin-bottom: 15px;
        flex-wrap: wrap;
    }
    .badge-score { 
        background: #ecfdf5; 
        color: #059669; 
        padding: 4px 10px; 
        border-radius: 20px; 
        font-weight: bold; 
        font-size: 0.85em;
        border: 1px solid #a7f3d0;
    }
    .badge-level {
        background: #f3f4f6;
        color: #374151;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.85em;
        font-weight: 600;
        border: 1px solid #e5e7eb;
    }
    .btn-link { 
        display: block; 
        text-align: center; 
        background: #0ea5e9; 
        color: white !important; 
        padding: 10px; 
        border-radius: 6px; 
        text-decoration: none; 
        font-weight: bold; 
        margin-top: auto; 
        transition: background 0.2s;
    }
    .btn-link:hover { background: #0284c7; }
</style>
""", unsafe_allow_html=True)

# --- 4. LOAD HỆ THỐNG ---
@st.cache_resource
def load_system():
    return CourseRecommender()

try:
    with st.spinner("🤖 Khởi động hệ thống..."):
        recsys = load_system()
except Exception as e:
    st.error(f"Lỗi: {e}")
    st.stop()

# --- 5. GIAO DIỆN ---
with st.sidebar:
    st.title("⚙️ Cấu Hình")
    yoe = st.slider("Kinh nghiệm (năm)", 0, 15, 2)
    mode = st.radio("Mục tiêu", ["quick", "deep"], format_func=lambda x: "⚡ Học nhanh" if x=="quick" else "🎓 Học sâu")
    skills_input = st.text_area("Kỹ năng đã có", "python, excel", height=100)
    known_skills = [s.strip() for s in skills_input.split(',') if s.strip()]

st.title("🎓 Lộ Trình Học Tập AI")

col1, col2 = st.columns([3, 1])
with col1:
    jd_text = st.text_area("📋 Dán nội dung tuyển dụng (JD) vào đây:", height=150)
with col2:
    st.write("")
    st.write("")
    analyze_btn = st.button("🚀 TẠO LỘ TRÌNH", type="primary", use_container_width=True)
    if st.button("🎲 JD Mẫu", use_container_width=True):
        jd_text = "AI Engineer with Python, Deep Learning, and Biology knowledge."
        st.rerun()

# --- 6. XỬ LÝ & HIỂN THỊ ---
if analyze_btn and jd_text:
    with st.spinner("Đang xử lý..."):
        time.sleep(0.5)
        
        # 1. Tự động lấy keyword
        auto_keywords = extract_keywords(jd_text)
        
        # 2. Chạy gợi ý
        user_profile = {'years_experience': yoe, 'learning_mode': mode, 'known_skills': known_skills}
        path_result = recsys.recommend(jd_text, user_profile, top_k=6, boost_keywords=auto_keywords)
        
        # 3. HIỂN THỊ KẾT QUẢ
        for stage_name, courses in path_result.items():
            if not courses: continue
            
            # Header Giai đoạn
            st.markdown(f"<div class='stage-header'>🚀 {stage_name}</div>", unsafe_allow_html=True)
            
            # Grid 3 cột
            cols = st.columns(3)
            for i, course in enumerate(courses):
                info = course['data']
                score = course['final_score']
                
                # Icon Level
                lvl_text = str(info.get('level', 'N/A'))
                lvl_icon = "🟢" if "beginner" in lvl_text.lower() else ("🔴" if "advanced" in lvl_text.lower() else "🟡")
                
                # [QUAN TRỌNG] Tạo HTML không thụt dòng để tránh lỗi hiển thị Code Block
                card_html = f"""
<div class='course-card'>
    <div class='course-title'>{info['course_name']}</div>
    <div class='badge-container'>
        <span class='badge-score'>{score:.0%} Match</span>
        <span class='badge-level'>{lvl_icon} {lvl_text}</span>
    </div>
    <a href="{info.get('page_url', '#')}" target="_blank" class="btn-link">
        Xem Khóa Học
    </a>
</div>
"""
                with cols[i % 3]:
                    st.markdown(card_html, unsafe_allow_html=True)

elif analyze_btn:
    st.warning("Vui lòng nhập JD.")