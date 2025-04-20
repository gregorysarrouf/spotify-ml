import streamlit as st
import numpy as np
import pandas as pd
import pickle
import joblib

st.set_page_config(page_title="Song Popularity Predictor", layout="centered")

st.title("🎶 Song Popularity Predictor")
st.markdown("Fill in the song's features below to predict its popularity!")

with st.expander("🎛️ Audio Features"):
    loudness = st.slider('Loudness', -60.0, 0.0, -10.0)
    energy = st.slider('Energy', 0.0, 1.0, 0.5)
    valence = st.slider('Valence', 0.0, 1.0, 0.5)
    danceability = st.slider('Danceability', 0.0, 1.0, 0.5)
    tempo = st.slider('Tempo', 40, 250, 120)
    duration_ms = st.number_input('Duration (ms)', 10000, 600000, 200000)
    liveness = st.slider('Liveness', 0.0, 1.0, 0.1)
    speechiness = st.slider('Speechiness', 0.0, 1.0, 0.05)
    acousticness = st.slider('Acousticness', 0.0, 1.0, 0.1)

with st.expander("🎼 Musical Features"):
    instrumental_loud_ratio = st.slider('Instrumental Loudness Ratio', 0.0, 1.0, 0.5)
    release_year_inverse = st.number_input('Release Year', 1950, 2025, 2020)

with st.expander("📦 Playlist & Metadata"):
    playlist_genre = st.text_input("Playlist Genre", "pop")
    playlist_subgenre = st.text_input("Playlist Subgenre", "mainstream")
    playlist_name = st.text_input("Playlist Name", "Today's Top Hits")
    track_artist = st.text_input("Track Artist", "Taylor Swift")

# Compute loud_dur_ratio and acousticness_inverse (if needed)
loud_dur_ratio = loudness / duration_ms
acousticness_inverse = 1.0 - acousticness

# Create raw input data
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

# Convert to DataFrame
df = pd.DataFrame([input_data])

# Load encoders
genre_encoder = joblib.load('playlist_genre_encoder.pkl')
subgenre_encoder = joblib.load('playlist_subgenre_encoder.pkl')
artist_encoder = joblib.load('track_artist_encoder.pkl')
name_encoder = joblib.load('playlist_name_encoder.pkl')

# Apply encoders
df['playlist_genre'] = genre_encoder.transform([df['playlist_genre'].iloc[0]])
df['playlist_subgenre'] = subgenre_encoder.transform([df['playlist_subgenre'].iloc[0]])
df['track_artist'] = artist_encoder.transform([df['track_artist'].iloc[0]])
df['playlist_name'] = name_encoder.transform([df['playlist_name'].iloc[0]])

# Reorder columns to match training set
expected_columns = [
    'energy', 'tempo', 'danceability', 'playlist_genre', 'loudness',
    'liveness', 'valence', 'track_artist', 'speechiness', 'playlist_name',
    'duration_ms', 'playlist_subgenre', 'acousticness_inverse',
    'instrumental_loud_ratio', 'loud_dur_ratio', 'release_year_inverse'
]

df = df[expected_columns]

model = joblib.load('popularity_predictor.pkl')
# Predict button
if st.button("🔍 Predict Popularity"):
    proba = model.predict_proba(df.to_numpy())[0]
    prob_of_popular = proba[1] 

    st.subheader("📈 Predicted Popularity:")
    st.write("Probability of Popularity:", round(prob_of_popular * 100, 2), "%")

    if prob_of_popular >= 0.5:
        st.success("🎉 This song is likely to be popular!")
    else:
        st.error("🙁 This song is unlikely to be popular.")


    st.markdown("### 🧾 Input Summary")
    st.write(input_data)
    st.write("Model classes:", model.classes_)
