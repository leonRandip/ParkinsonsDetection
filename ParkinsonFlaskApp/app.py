import os
from flask import Flask, render_template, request
import joblib
import librosa
import numpy as np

app = Flask(__name__)

# Hardcode or load your classification accuracies from somewhere:
model_accuracies = {
    "Naive Bayes": 0.77,
    "SVM": 0.73,
    "Random Forest": 0.94,
    "Logistic Regression": 0.72,
    "K-NN": 0.61
}

# Load the Random Forest model from the 'model' folder
try:
    rf_model = joblib.load("./model/random_forest_model.pkl")
    print("RandomForest model loaded successfully!")
except Exception as e:
    rf_model = None
    print(f"Failed to load RandomForest model: {e}")

@app.route('/')
def home():
    return render_template('home.html', accuracies=model_accuracies)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # 1) Check if 'audio_file' is part of request
        if 'audio_file' not in request.files:
            return "No file part in request"
        audio_file = request.files['audio_file']

        # 2) Verify the user actually selected a file
        if audio_file.filename == '':
            return "No selected file"

        # 3) Load the audio with librosa
        try:
            y, sr = librosa.load(audio_file, sr=None)
        except Exception as e:
            return f"Error loading audio: {e}"

        # ----- EXTRACT FEATURES (example) -----
        f0, voiced_flag, voiced_prob = librosa.pyin(y, sr=sr, fmin=50, fmax=300)
        f0_clean = np.nan_to_num(f0)

        # Naive example for jitter, shimmer, mean_f0
        jitter = np.std(f0_clean)
        shimmer = np.std(y)
        mean_f0 = np.mean(f0_clean)

        # Build feature vector
        features = np.array([jitter, shimmer, mean_f0]).reshape(1, -1)

        # 4) Predict
        if rf_model is None:
            return "No model loaded!"
        prediction = rf_model.predict(features)
        label = "Positive for Parkinson's" if prediction[0] == 1 else "Negative for Parkinson's"

        return render_template(
            "predict.html",
            prediction_text=f"Prediction: {label}",
            jitter_val=f"{jitter:.4f}",
            shimmer_val=f"{shimmer:.4f}",
            mean_f0_val=f"{mean_f0:.4f}"
        )
    else:
        # GET -> just render the predict page
        return render_template("predict.html")

if __name__ == "__main__":
    # Optional fallback for local dev: read PORT from environment or default to 5000
    port = int(os.environ.get("PORT", 5000))
    # app.run in debug=False for production, but you can set True if you want local debugging
    app.run(host="0.0.0.0", port=port, debug=False)
