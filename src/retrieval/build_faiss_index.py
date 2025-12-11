import faiss
import numpy as np
import os
import argparse
import pickle

def build_index(emb_path, index_out_path):
    print(f"Loading embeddings from {emb_path}...")
    if not os.path.exists(emb_path):
        raise FileNotFoundError(f"File not found: {emb_path}")
        
    embeddings = np.load(emb_path).astype('float32')
    d = embeddings.shape[1]
    print(f"Vector dimension: {d}, Count: {embeddings.shape[0]}")
    
    # Sử dụng Inner Product (IP) cho Cosine Similarity (nếu vector đã normalize)
    # Hoặc L2 nếu muốn tìm theo khoảng cách Euclid
    # Với SBERT, thường dùng IP (tương đương Cosine nếu norm=1)
    index = faiss.IndexFlatIP(d)
    
    print("Training/Adding vectors to Faiss index...")
    index.add(embeddings)
    
    # Tạo thư mục
    os.makedirs(os.path.dirname(index_out_path), exist_ok=True)
    
    # Lưu index
    faiss.write_index(index, index_out_path)
    print(f"Saved Faiss index to: {index_out_path}")

if __name__ == "__main__":
    BASE_DIR = "/Users/nhitruong/Documents/data_mining_project"
    parser = argparse.ArgumentParser()
    # Lưu ý: Dùng vector GỐC (768 chiều) để tìm kiếm chính xác nhất, PCA chỉ dùng cho Clustering
    parser.add_argument("--emb", default=f"{BASE_DIR}/outputs/embeddings/course_emb.npy")
    parser.add_argument("--index", default=f"{BASE_DIR}/outputs/indices/faiss_index.bin")
    
    args = parser.parse_args()
    build_index(args.emb, args.index)