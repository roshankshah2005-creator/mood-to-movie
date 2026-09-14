from fastapi import FastAPI,HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import requests
import os
from typing import List

app=FastAPI()

df=pd.read_csv("data/processed/clean_movies.csv")
movie_embeddings=np.load("data/processed/movie_embeddings.npy")

API_URL="https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2"
HF_TOKEN=os.environ.get("HF_TOKEN")

class Query(BaseModel):
    vector : List[float]
    num_results : int=5

@app.post("/api/recommend")
def recommend(query : Query):
    try:
        user_vector = np.array([query.vector])
        
        similarities = cosine_similarity(user_vector, movie_embeddings)[0]
        top_indices = similarities.argsort()[::-1][:query.num_results]
        
        results = []
        for idx in top_indices:
            results.append({
                "title": str(df.iloc[idx]["title"]),
                "genres": str(df.iloc[idx]["genres"]),
                "overview": str(df.iloc[idx]["overview"]),
                "score": round(float(similarities[idx]) * 100, 1)
            })
            
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))