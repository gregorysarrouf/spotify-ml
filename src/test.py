import pandas as pd
import joblib

# The final values for a single track
test_values = {
    'energy': 0.592,
    'tempo': 157.969,
    'danceability': 0.521,
    'playlist_genre': 25,
    'loudness': -7.777,
    'liveness': 0.122,
    'valence': 0.535,
    'track_artist': 1607,  # Assuming this is already encoded
    'speechiness': 0.0304,
    'playlist_name': 104,
    'duration_ms': 251668.0,
    'playlist_subgenre': 55,
    'acousticness_inverse': 0.6920,
    'instrumental_loud_ratio': -0.000000,
    'loud_dur_ratio': -0.000031,
    'release_year_inverse': 0.000494
}

# Create a test DataFrame with these values
test_df = pd.DataFrame([test_values])
model = joblib.load('popularity_predictor.pkl')
prediction = model.predict(test_df)  # this still uses default threshold (0.5)

prob = model.predict_proba(test_df)[0][1]  # prob of class 1
prediction = 1 if prob > 0.2 else 0  # this line overrides the above prediction

print("Adjusted Prediction:", prediction)  # ✅ this will be based on your 0.3 threshold

print("\nFinal Test Data (Prepared for Prediction):")
print(test_df)
print(prediction)  # ✅ this is still your adjusted prediction


