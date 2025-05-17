import streamlit as st
import pandas as pd
import numpy as np
import ast
from surprise import Dataset, Reader, SVD
import pickle
from pathlib import Path

st.set_page_config(page_title="🎬 Film Öneri Sistemi", layout="wide", page_icon="🎬")
st.title("🎬 Film Öneri Sistemi")

@st.cache_data
def load_data():
    ratings = pd.read_csv("data/ratings_small.csv")
    movies = pd.read_csv("data/movies_metadata.csv", low_memory=False)
    movies = movies[movies['id'].apply(lambda x: str(x).isdigit())]
    movies['id'] = movies['id'].astype(int)
    return ratings, movies

ratings, movies = load_data()

@st.cache_data
def process_movies(movies):
    def parse_genres(genre_str):
        try:
            genres = ast.literal_eval(genre_str)
            return [genre['name'] for genre in genres]
        except:
            return []
    movies['genres'] = movies['genres'].apply(parse_genres)
    return movies

movies = process_movies(movies)

@st.cache_data
def load_credits():
    credits = pd.read_csv("data/credits.csv")
    credits['cast'] = credits['cast'].apply(lambda x: ast.literal_eval(x) if pd.notnull(x) else [])
    credits['cast_names'] = credits['cast'].apply(lambda x: [d['name'] for d in x])
    credits['id'] = credits['id'].astype(int)
    return credits

credits = load_credits()

model_path = "data/svd_model.pkl"
if Path(model_path).exists():
    with open(model_path, "rb") as f:
        model = pickle.load(f)
else:
    reader = Reader(rating_scale=(0.5, 5.0))
    data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
    trainset = data.build_full_trainset()
    model = SVD()
    model.fit(trainset)
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

genres_list = sorted({genre for sublist in movies['genres'] for genre in sublist})

if 'genre_weights' not in st.session_state:
    st.session_state.genre_weights = {g: 3 for g in genres_list}
    st.session_state.show_modal = True

if st.session_state.get('show_modal', True):
    st.markdown("## 🎯 Tür Tercihlerinizi Belirtin")
    st.info("Uygulamayı kullanmadan önce hangi türleri ne kadar sevdiğinizi belirtin.")
    with st.form("genre_modal_form"):
        new_weights = {}
        cols = st.columns(3)
        for i, genre in enumerate(genres_list):
            col = cols[i % 3]
            with col:
                new_weights[genre] = st.slider(genre, 1, 5, 3)
        if st.form_submit_button("✔️ Tamam"):
            st.session_state.genre_weights = new_weights
            st.session_state.show_modal = False
    st.stop()

genre_weights = st.session_state.get('genre_weights', {})

st.markdown("## 🎛️ Arama")
query = st.text_input("🔍 Film, Oyuncu veya Tür Ara", placeholder="Örnek: Inception, Brad Pitt, Drama")
recommend_button = st.button("🎯 Ara ve Öner")

if recommend_button and query:
    keywords = [q.strip().lower() for q in query.split(',')]

    film_matches = movies[movies['title'].fillna('').str.lower().apply(lambda t: any(k in t for k in keywords))]
    cast_matches = credits[credits['cast_names'].apply(lambda names: any(any(k in name.lower() for name in names) for k in keywords))]
    cast_merged = cast_matches.merge(movies, left_on='id', right_on='id')
    genre_matches = movies[movies['genres'].apply(lambda genres: any(any(k in genre.lower() for genre in genres) for k in keywords))]

    merged = pd.concat([film_matches, cast_merged, genre_matches]).drop_duplicates(subset='id').head(100)

    movie_avg_ratings = ratings.groupby('movieId')['rating'].mean().reset_index()
    movie_avg_ratings.columns = ['id', 'predicted_rating']
    merged = merged.merge(movie_avg_ratings, on='id', how='left')
    merged['predicted_rating'] = merged['predicted_rating'].fillna(3.0)

    genre_scores = merged['genres'].apply(lambda g: np.mean([genre_weights.get(x, 3) for x in g]))
    merged['predicted_rating'] = (merged['predicted_rating'] + genre_scores) / 2

    if not merged.empty:
        st.subheader("🎯 Arama Sonuçları ve Tahmini Puanlar")
        for _, row in merged.sort_values(by='predicted_rating', ascending=False).head(10).iterrows():
            cast_list = credits[credits['id'] == row['id']]['cast_names']
            cast_str = ', '.join(cast_list.values[0][:5]) if not cast_list.empty else 'Bilinmiyor'
            poster_url = f"https://image.tmdb.org/t/p/w200{row['poster_path']}" if pd.notna(row.get('poster_path')) else ""
            st.markdown(f"""
            <div style='display:flex; gap:15px; align-items:center; background-color:#0e1117; padding:10px; margin-bottom:10px; border-radius:10px; border-left: 5px solid #00bcd4;'>
                <img src='{poster_url}' style='height:150px; border-radius:5px;' alt='Poster'>
                <div style='color:white;'>
                    <h4 style='margin-bottom:5px;'>🎬 {row['title']}</h4>
                    <p>🌟 Tahmini Puan: <strong>{row['predicted_rating']:.2f}</strong><br>
                    🎞️ Türler: {', '.join(row['genres'])}<br>
                    👥 Oyuncular: {cast_str}</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.warning("Aramanızla eşleşen sonuç bulunamadı.")