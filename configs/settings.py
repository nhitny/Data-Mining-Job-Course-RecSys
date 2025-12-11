import os
from pathlib import Path
from dotenv import load_dotenv

# 1. Xác định đường dẫn gốc của dự án
# (Đi từ: src/config/settings.py -> src/config -> src -> data_mining_project)
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# 2. Đường dẫn đến file .env
ENV_PATH = BASE_DIR / ".env"

# 3. Load biến môi trường
if ENV_PATH.exists():
    load_dotenv(dotenv_path=ENV_PATH)
    print(f"Configuration loaded from: {ENV_PATH}")
else:
    print(f"Warning: .env file not found at {ENV_PATH}")

# 4. Export các biến cấu hình
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Các cấu hình khác (nếu có sau này)
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"