import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# --- 1. Page Configuration ---
st.set_page_config(page_title="Mood-to-Movie Matcher", page_icon="🎬", layout="centered")

# --- 2. Caching Data (Loads once, stays fast) ---
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_data
def load_data():
    df = pd.read_csv("data/processed/clean_movies.csv")
    embeddings = np.load("data/processed/movie_embeddings.npy")
    return df, embeddings

st.title("🎬 Semantic Mood-to-Movie Matcher")
st.markdown("Forget genres. Describe exactly what kind of **vibe, mood, or feeling** you want to watch.")

# Load our assets
with st.spinner("Loading AI model and vectors..."):
    model = load_model()
    df, movie_embeddings = load_data()

# --- 3. User Interface ---
user_mood = st.text_area(
    "What are you in the mood for?", 
    placeholder="e.g., A rainy cyberpunk detective story that feels lonely but action-packed.",
    height=100
)

num_results = st.slider("How many recommendations?", min_value=1, max_value=10, value=5)

# --- 4. The Matching Logic ---
if st.button("Find My Movie"):
    if user_mood.strip() == "":
        st.warning("Please enter a mood first!")
    else:
        with st.spinner("Calculating semantic match..."):
            
            query_vector = model.encode([user_mood])
            
            similarities = cosine_similarity(query_vector, movie_embeddings)[0]
            
            top_indices = similarities.argsort()[::-1][:num_results]
            
            # --- 5. Display Results ---
            st.divider()
            st.subheader("Here is what you should watch:")
            
            for rank, idx in enumerate(top_indices, 1):
                match_score = round(similarities[idx] * 100, 1)
                title = df.iloc[idx]["title"]
                genres = df.iloc[idx]["genres"]
                overview = df.iloc[idx]["overview"]
                
                with st.container():
                    st.markdown(f"### #{rank}: {title}")
                    st.markdown(f"**Match Score:** `{match_score}%` | **Genres:** {genres}")
                    st.write(overview)
                    st.markdown("---")