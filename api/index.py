from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import csv
from typing import List

app = FastAPI()

# 1. Load CSV using Python's built-in CSV reader (Zero megabytes added!)
movies = []
with open("data/processed/clean_movies.csv", mode="r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        movies.append(row)

# 2. Load the vectors
movie_embeddings = np.load("data/processed/movie_embeddings.npy")

class Query(BaseModel):
    vector: List[float]
    num_results: int = 5

@app.post("/api/recommend")
def recommend(query: Query):
    try:
        user_vector = np.array(query.vector)
        
        # 3. Calculate Cosine Similarity using pure numpy
        dot_products = np.dot(movie_embeddings, user_vector)
        norm_movies = np.linalg.norm(movie_embeddings, axis=1)
        norm_user = np.linalg.norm(user_vector)
        
        similarities = dot_products / (norm_movies * norm_user + 1e-10)
        
        # 4. Get the top matches
        top_indices = similarities.argsort()[::-1][:query.num_results]
        
        results = []
        for idx in top_indices:
            results.append({
                "title": movies[idx]["title"],
                "genres": movies[idx]["genres"],
                "overview": movies[idx]["overview"],
                "score": round(float(similarities[idx]) * 100, 1)
            })
            
        return {"results": results}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
