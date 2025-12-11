import time
import numpy as np
# Giả sử bạn import các module thật
# from src.retrieval.semantic_search import search_faiss
# from src.clustering.cluster_search import search_cluster

def compare_methods(jd_vectors, ground_truths):
    """
    Input:
        - jd_vectors: List các vector embedding của JD
        - ground_truths: Kết quả mong đợi
    """
    results = {
        "faiss": {"time": [], "precision": []},
        "cluster": {"time": [], "precision": []}
    }
    
    print("Comparing Faiss vs Clustering...")
    
    for i, vec in enumerate(jd_vectors):
        # 1. Test Faiss (Brute-force or IVF)
        start = time.time()
        # faiss_res = search_faiss(vec, top_k=10) # CALL REAL FUNCTION
        faiss_time = time.time() - start
        
        # 2. Test Cluster (Predict cluster -> Search in cluster)
        start = time.time()
        # cluster_res = search_cluster(vec, top_k=10) # CALL REAL FUNCTION
        cluster_time = time.time() - start
        
        # Record Time
        results["faiss"]["time"].append(faiss_time)
        results["cluster"]["time"].append(cluster_time)
        
        # Record Precision (Mock logic)
        # p_faiss = calculate_precision(faiss_res, ground_truths[i])
        # p_cluster = calculate_precision(cluster_res, ground_truths[i])
    
    avg_time_faiss = np.mean(results["faiss"]["time"])
    avg_time_cluster = np.mean(results["cluster"]["time"])
    
    print(f"Avg Time Faiss: {avg_time_faiss:.4f}s")
    print(f"Avg Time Cluster: {avg_time_cluster:.4f}s")
    
    improvement = (avg_time_faiss - avg_time_cluster) / avg_time_faiss * 100
    print(f"Speed Improvement: {improvement:.2f}%")