import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def load_data():
    """
    Loads the parkinson_speech_dataset.csv from the same directory
    or wherever you store it. Adjust as needed for your path.
    """
    file_path = "parkinson_speech_dataset.csv"  # Adjust path if needed

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    data = pd.read_csv(file_path)
    print(f"\nLoaded dataset with {len(data)} samples.")
    print("Columns:", data.columns.tolist())
    print("Class distribution:\n", data['Class Status'].value_counts())

    return data

def prepare_data(data):
    """
    Splits the DataFrame into train/test sets, using just 3 columns:
    Jitter(%), Shimmer(dB), HNR
    """
    # 1) Define which columns to keep
    # Make sure these exactly match your CSV's column names.
    feature_cols = ['Jitter(%)', 'Shimmer(dB)', 'HNR']
    
    # 2) Separate X (features) and y (labels)
    X = data[feature_cols]
    y = data['Class Status']

    # 3) Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples:  {len(X_test)}")

    return X_train, X_test, y_train, y_test

def train_and_save_random_forest(X_train, X_test, y_train, y_test):
    """
    Trains a RandomForestClassifier on the 3 columns and saves it as .pkl.
    """
    rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)

    print("\nTraining Random Forest on 3 features: [Jitter(%), Shimmer(dB), HNR]")
    rf_clf.fit(X_train, y_train)

    y_pred = rf_clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Random Forest Accuracy: {accuracy:.4f}")

    # Save the model
    model_path = "random_forest_model.pkl"
    joblib.dump(rf_clf, model_path)
    print(f"Model saved to: {model_path}")

def main():
    print("Starting Random Forest Training Script (3 features)...")
    data = load_data()
    X_train, X_test, y_train, y_test = prepare_data(data)
    train_and_save_random_forest(X_train, X_test, y_train, y_test)
    print("\nDone!")

if __name__ == "__main__":
    main()
