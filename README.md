# 🎓 Data-Mining-Job-Course-RecSys (CS2207 - Hệ thống Gợi ý Lộ trình Học tập)

## 💡 Giới Thiệu
Dự án này ứng dụng các kỹ thuật Khai thác Dữ liệu (Data Mining) và Deep Learning (NLP) để xây dựng một hệ thống gợi ý khóa học **cá nhân hóa** (Personalized) dựa trên yêu cầu công việc (Job Description - JD) và hồ sơ kinh nghiệm cá nhân.

Hệ thống chuyển đổi một danh sách khóa học truyền thống thành một **Lộ trình học tập có cấu trúc (Learning Path)**, phù hợp với trình độ và mục tiêu nghề nghiệp của từng người dùng.

## ✨ 1. Tính Năng Cốt Lõi và Giá Trị Ứng Dụng

| Tính năng | Công nghệ | Giá trị mang lại |
| :--- | :--- | :--- |
| **Semantic Matching** | **SBERT** (Sentence-BERT) | Hiểu ý nghĩa JD, không bỏ sót khóa học. |
| **Deep Personalization** | **YoE Mapping** (Năm kinh nghiệm) | Lọc bỏ khóa **Beginner** cho Senior (tránh lãng phí thời gian). |
| **Domain Boosting** | **Rule-based Weights** | Ưu tiên các khóa học chứa từ khóa chuyên môn hẹp (Genomics, FinTech). |
| **Learning Path** | **Stage Grouping** | Sắp xếp kết quả thành 3 Giai đoạn (Foundation, Core, Advanced). |
| **Topic Discovery** | **K-Means Clustering** | Phân cụm các khóa học để chứng minh Categories gốc bị chồng chéo. |
| **Evaluation** | **LLM-as-a-Judge** | Sử dụng AI (Gemini) để kiểm định tính chính xác của gợi ý. |

## 🛠️ 2. Hướng Dẫn Cài Đặt và Vận Hành

### Bước 1: Chuẩn bị Môi trường
1.  **Clone Repository:**
    ```bash
    git clone https://github.com/nhitny/Data-Mining-Job-Course-RecSys
    cd Data-Mining-Job-Course-RecSys
    ```
2.  **Cài đặt Thư viện:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Đặt Dữ liệu:** Đặt file thô (`coursera_courses.csv`, `linkedin_jobs.csv`) vào thư mục **`data/raw/`**.
4.  **Cấu hình API:** Thêm Gemini API Key vào file `src/data_labeling.py` và `src/evaluation.py` để chạy bước kiểm chứng.

### Bước 2: Chạy Pipeline Xử Lý Dữ liệu (The Core Chain)
Bạn phải chạy tuần tự các script Python sau:

| STT | Script | Mục đích |
| :--- | :--- | :--- |
| **1** | `python src/data_cleaning.py` | **Tiền xử lý:** Làm sạch Text, chuẩn hóa cột Level/Rating. |
| **2** | `python src/embedding.py` | **Mã hóa Vector (SBERT)** cho toàn bộ khóa học. |
| **3** | `python src/3_clustering.py` | **Phân cụm K-Means** và gán nhãn Cluster vào dữ liệu. |

### Bước 3: Vận hành & Kiểm chứng Hệ thống

| File | Mục đích |
| :--- | :--- |
| `python src/data_labeling.py` | **Tạo Ground Truth:** Dùng Gemini chấm điểm 20 JD mẫu (Bước chuẩn bị cho Evaluation). |
| `python src/evaluation.py` | **Tính Metrics:** Tính Precision, NDCG, MRR, và tạo biểu đồ báo cáo. |
| `streamlit run src/app_ui.py` | **Chạy Giao diện Web App** (Demo cuối cùng). |

---
**Tác giả:** [Tên bạn hoặc Nhóm bạn] 

**Môn học:** CS2207 - Khai thác dữ liệu và Ứng dụng