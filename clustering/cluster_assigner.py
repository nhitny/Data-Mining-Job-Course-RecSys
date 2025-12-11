import numpy as np
import pickle
import os

class ClusterAssigner:
    def __init__(self, pca_path, kmeans_path):
        """
        Load PCA và KMeans model đã train để gán cụm cho vector mới.
        """
        # Kiểm tra file tồn tại
        if not os.path.exists(pca_path):
            raise FileNotFoundError(f"PCA Model not found at: {pca_path}")
        if not os.path.exists(kmeans_path):
            raise FileNotFoundError(f"KMeans Model not found at: {kmeans_path}")
            
        # Load PCA
        with open(pca_path, 'rb') as f:
            self.pca = pickle.load(f)
            
        # Load KMeans
        with open(kmeans_path, 'rb') as f:
            self.kmeans = pickle.load(f)
            
        print(f"   - Cluster Assigner Loaded (PCA + KMeans)")

    def get_cluster(self, embedding_vector):
        """
        Input: Vector gốc (768 chiều)
        Output: Cluster ID (int)
        """
        # 1. Reshape nếu là 1D array (768,) -> (1, 768)
        if len(embedding_vector.shape) == 1:
            embedding_vector = embedding_vector.reshape(1, -1)
            
        # 2. Giảm chiều bằng PCA đã train
        # Lưu ý: Phải dùng đúng PCA đã học trên tập train, không fit lại
        pca_vector = self.pca.transform(embedding_vector)
        
        # 3. Dự đoán cụm bằng KMeans đã train
        cluster_id = self.kmeans.predict(pca_vector)[0]
        
        return int(cluster_id)