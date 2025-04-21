import pandas as pd
import joblib

# Updated values for a single track
test_values = {
    'energy': 0.424,
    'tempo': 75.466,
    'danceability': 0.352,
    'playlist_genre': 15,
    'loudness': -8.009,
    'liveness': 0.2420,
    'valence': 0.605,
    'track_artist': 1769,  # Assuming this is already encoded
    'speechiness': 0.0634,
    'playlist_name': 23,
    'duration_ms': 661293.0,
    'playlist_subgenre': 23,
    'acousticness_inverse': 0.035,
    'instrumental_loud_ratio': -0.000000,
    'loud_dur_ratio': -0.000012,
    'release_year_inverse': 0.000497
}

# Create test DataFrame
test_df = pd.DataFrame([test_values])

# Load model
model = joblib.load('popularity_predictor.pkl')
prediction = model.predict(test_df)

print("OG Prediction:", prediction)
# Make prediction
prob = model.predict_proba(test_df)[0][1]  # Probability of class 1
prediction = 1 if prob > 0.27 else 0       # Custom threshold at 0.27

# Display results
print("Adjusted Prediction:", prediction)
