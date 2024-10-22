import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Preprocessing the datasetcd
def load_data(file_path):
    data = pd.read_csv('ht.csv')
    X = data.iloc[:, :-1]  # Features
    y = data.iloc[:, -1]   # Labels (Malicious/Benign)
    return X, y

# Train the model
def train_model(X, y, model_type="random_forest"):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    if model_type == "random_forest":
        model = RandomForestClassifier(n_estimators=100)
    elif model_type == "decision_tree":
        model = DecisionTreeClassifier()
    elif model_type == "svm":
        model = SVC(kernel='linear')
    else:
        raise ValueError("Unsupported model type. Choose from 'random_forest', 'decision_tree', or 'svm'")
    
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    
    print(f"Model: {model_type}")
    print(f"Accuracy: {accuracy_score(y_test, predictions)}")
    print("Classification Report:\n", classification_report(y_test, predictions))
    
    return model

# Save the trained model
def save_model(model, output_file):
    joblib.dump(model, output_file)
    print(f"Model saved as {output_file}")

# Load the saved model and classify new data
def detect_intrusion(model_file, test_data_file):
    model = joblib.load(model_file)
    X_test = pd.read_csv(test_data_file)
    predictions = model.predict(X_test)
    print("Predictions: ", predictions)

# Main CLI handler
def main():
    parser = argparse.ArgumentParser(description="Intrusion Detection System using Machine Learning")
    
    subparsers = parser.add_subparsers(dest="command")
    
    # Train model command
    train_parser = subparsers.add_parser('train', help='Train the model')
    train_parser.add_argument('dataset', type=str, help='Path to the dataset file')
    train_parser.add_argument('--model', type=str, default="random_forest", choices=["random_forest", "decision_tree", "svm"], help="Model to train (default: random_forest)")
    train_parser.add_argument('--output', type=str, default="model.joblib", help="Output file for saving the model")
    
    # Detect intrusion command
    detect_parser = subparsers.add_parser('detect', help='Detect intrusion using saved model')
    detect_parser.add_argument('model', type=str, help='Path to the saved model file')
    detect_parser.add_argument('test_data', type=str, help='Path to the test data file')

    args = parser.parse_args()
    
    if args.command == "train":
        X, y = load_data(args.dataset)
        model = train_model(X, y, args.model)
        save_model(model, args.output)
    elif args.command == "detect":
        detect_intrusion(args.model, args.test_data)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
