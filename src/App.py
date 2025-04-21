import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Page setup
st.set_page_config(page_title="Song Popularity Predictor", layout="centered")

st.title("🎶 Song Popularity Predictor")
st.markdown("Fill in the song's features below to predict its popularity!")

# Load encoders and model
genre_encoder = joblib.load('playlist_genre_encoder.pkl')
subgenre_encoder = joblib.load('playlist_subgenre_encoder.pkl')
artist_encoder = joblib.load('track_artist_encoder.pkl')
name_encoder = joblib.load('playlist_name_encoder.pkl')
model = joblib.load('popularity_predictor.pkl')

# Define the expected feature order
expected_columns = [
    'energy', 'tempo', 'danceability', 'playlist_genre', 'loudness',
    'liveness', 'valence', 'track_artist', 'speechiness', 'playlist_name',
    'duration_ms', 'playlist_subgenre', 'acousticness_inverse',
    'instrumental_loud_ratio', 'loud_dur_ratio', 'release_year_inverse'
]

# Input mode toggle
use_sample = st.checkbox("🎵 Use a sample song instead of manual input")

if use_sample:
    sample_df = pd.read_excel("../data/sample_test.xlsx")
    selected_song = st.selectbox("Select a sample song", sample_df['song_name'].unique())
    df = sample_df[sample_df['song_name'] == selected_song].drop(columns=["song_name", "popularity_class"]).reset_index(drop=True)
else:
    # Manual input mode
    with st.expander("🎛️ Audio Features"):
        loudness = st.slider('Loudness', -50.0, 0.0, -7.7)
        energy = st.number_input('Energy', 0.0, 1.0, 0.592, step=0.001, format="%.4f")
        valence = st.number_input('Valence', 0.0, 1.0, 0.535, step=0.001, format="%.4f")
        danceability = st.number_input('Danceability', 0.0, 1.0, 0.521, step=0.001, format="%.4f")
        tempo = st.slider('Tempo', 40.0, 250.0, 157.9)
        duration_ms = st.number_input('Duration (ms)', 10000, 1000000, 251668)
        liveness = st.number_input('Liveness', 0.0, 1.0, 0.122, step=0.001, format="%.4f")
        speechiness = st.number_input('Speechiness', 0.0, 1.0, 0.034, step=0.001, format="%.4f")
        acousticness = st.number_input('Acousticness', 0.0, 1.0, 0.0648, step=0.001, format="%.4f")

    with st.expander("🎼 Musical Features"):
        instrumentalness = st.number_input('Instrumentalness', 0.0, 1.0, 0.0, step=0.001, format="%.4f")
        release_year = st.number_input('Release Year', 1950, 2025, 2020)

    with st.expander("📦 Playlist & Metadata"):
        playlist_genre = st.text_input("Playlist Genre", "pop")
        playlist_subgenre = st.text_input("Playlist Subgenre", "mainstream")
        playlist_name = st.text_input("Playlist Name", "Today's Top Hits")
        track_artist = st.text_input("Track Artist", "Lady Gaga, Bruno Mars")

    # Derived features
    loud_dur_ratio = float(loudness) / float(duration_ms)
    acousticness_inverse = 1.0 - float(acousticness)
    instrumental_loud_ratio = float(instrumentalness) / float(loudness)
    release_year_inverse = 1.0 / float(release_year)

    # Construct DataFrame
    input_data = {
        "energy": energy,
        "tempo": tempo,
        "danceability": danceability,
        "playlist_genre": playlist_genre,
        "loudness": loudness,
        "liveness": liveness,
        "valence": valence,
        "track_artist": track_artist,
        "speechiness": speechiness,
        "playlist_name": playlist_name,
        "duration_ms": duration_ms,
        "playlist_subgenre": playlist_subgenre,
        "acousticness_inverse": acousticness_inverse,
        "instrumental_loud_ratio": instrumental_loud_ratio,
        "loud_dur_ratio": loud_dur_ratio,
        "release_year_inverse": release_year_inverse,
    }

    df = pd.DataFrame([input_data])

    # Encode categorical values
    df['playlist_genre'] = genre_encoder.transform([df['playlist_genre'].iloc[0]])
    df['playlist_subgenre'] = subgenre_encoder.transform([df['playlist_subgenre'].iloc[0]])
    df['track_artist'] = artist_encoder.transform([df['track_artist'].iloc[0]])
    df['playlist_name'] = name_encoder.transform([df['playlist_name'].iloc[0]])

# Ensure column order matches model
df = df[expected_columns]

# Prediction section
if st.button("🔍 Predict Popularity"):
    with st.spinner('Predicting...'):
        proba = model.predict_proba(df.to_numpy())[0]
        prob_of_popular = proba[1]

    # Prediction interpretation
    threshold = 0.265
    if prob_of_popular >= threshold:
        st.success(f"🎉 This song is likely to be **popular**! (Probability: {prob_of_popular:.2%})")
    else:
        st.error(f"🙁 This song is likely **not** popular. (Probability: {prob_of_popular:.2%})")

    # Input summary
    st.markdown("### 🧾 Input Summary")
    st.write(df)
