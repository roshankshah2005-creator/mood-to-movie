import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import time

print("Loading cleaned movie data...")
df=pd.read_csv("data/processed/clean_movies.csv")

texts=df["combined_features"].tolist()

print(f"Loaded {len(texts)} movies.")
print("Downloading/Loading the sentence Transformer model...")

model=SentenceTransformer("all-MiniLM-L6-v2")

print("Generating embeddings.This may take few minutes...")
start_time=time.time()

embeddings=model.encode(texts,show_progress_bar=True)

end_time=time.time()
print(f"Finished embedding in{round((end_time - start_time)/60 , 2)} minutes.")
print("Saving Vectors to disk.")

np.save("data/processed/movie_embeddings.npy",embeddings)

print("Success! Embeddings saved to 'data/processed/movie_embeddings.npy'.")