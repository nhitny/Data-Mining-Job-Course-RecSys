import re

class JDCleaner:
    def __init__(self):
        pass

    def clean(self, text):
        if not isinstance(text, str):
            return ""
        
        # 1. Chuyển về chữ thường
        text = text.lower()
        
        # 2. Loại bỏ email
        text = re.sub(r'\S+@\S+', '', text)
        
        # 3. Loại bỏ đường dẫn URL
        text = re.sub(r'http\S+', '', text)
        
        # 4. Giữ lại các ký tự chữ (bao gồm tiếng Việt), số và các ký tự đặc biệt quan trọng (+, #, .)
        # \w khớp với chữ cái (Unicode) và số
        text = re.sub(r'[^\w\s\+\#\.]', ' ', text)
        
        # 5. Xóa khoảng trắng thừa
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text

# Phần test chạy trực tiếp trong file này (không cần import)
if __name__ == "__main__":
    cleaner = JDCleaner()
    
    # Test mẫu JD tiếng Anh
    raw_jd_english = """
    ***URGENT HIRING***: Senior Backend Engineer (Remote) 🚀
    Contact: careers@tech.com | Req: Python 3.9, C++, .NET Core
    """
    
    print("-" * 50)
    print("ORIGINAL:")
    print(raw_jd_english)
    print("-" * 50)
    print("CLEANED:")
    print(cleaner.clean(raw_jd_english))