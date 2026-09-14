import pandas as pd

print("Loading 1M dataset...")

df=pd.read_csv("data/raw/TMDB_movie_dataset_v11.csv",low_memory=False)

print(f"Original dataset size:{len(df)} movies")

df=df.dropna(subset=["overview"])

if "vote_count" in df.columns:
    df=df[df["vote_count"]>50]

    if "popularity" in df.columns:
        df=df.sort_values(by="popularity",ascending=False)

df=df.head(100000).copy()
print(f"Filtered dataset size: {len(df)} movies")

df["genres"]=df["genres"].fillna("")
df["combined_features"]=df["title"]+"."+df["genres"]+"."+df["overview"]

clean_df=df[["title","overview","genres","combined_features"]]

clean_df.to_csv("data/processed/clean_movies.csv",index=False)
print("Saved clean dataset as 'clean_movies.csv'.")