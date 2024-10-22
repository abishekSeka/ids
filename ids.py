# Required libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

# Constants
USER_DATA_FILE = "users.txt"

# Load dataset
def load_data(file_path):
    """Load dataset from a CSV file."""
    try:
        data = pd.read_csv(file_path)
        return data
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        st.error(f"Error loading data: {e}")
        return None

# Save the trained model
def save_model(model, filename):
    """Save the trained model to a file."""
    joblib.dump(model, filename)

# Plot confusion matrix
def plot_confusion_matrix(y_true, y_pred, model_name):
    """Plot and save the confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Benign', 'Malicious'], yticklabels=['Benign', 'Malicious'])
    plt.title(f'Confusion Matrix: {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(f'confusion_matrix_{model_name}.png')
    plt.close()

# Plot ROC curve
def plot_roc_curve(y_true, y_prob, model_name):
    """Plot and save the ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC Curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='red', linestyle='--')
    plt.title(f'ROC Curve: {model_name}')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(f'roc_curve_{model_name}.png')
    plt.close()

# Train model based on selection
def train_model(X, y, model_name):
    """Train the model based on user's selection."""
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "SVM": SVC(probability=True, random_state=42)
    }

    model = models[model_name]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]  # Probability for ROC curve

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    return model, accuracy, report, y_pred, y_prob

# User registration
def register_user(username, password):
    """Register a new user."""
    if os.path.exists(USER_DATA_FILE):
        with open(USER_DATA_FILE, "r") as file:
            if any(user.split(":")[0] == username for user in file):
                st.error("Username already exists. Please choose a different username.")
                return False
    
    with open(USER_DATA_FILE, "a") as file:
        file.write(f"{username}:{password}\n")
    st.success("Registration successful!")
    return True

# User login
def login_user(username, password):
    """Log in an existing user."""
    if not os.path.exists(USER_DATA_FILE):
        st.error("No registered users found. Please register first.")
        return False

    with open(USER_DATA_FILE, "r") as file:
        for line in file:
            stored_username, stored_password = line.strip().split(":")
            if stored_username == username and stored_password == password:
                st.success("Login successful!")
                return True
    st.error("Login failed! Incorrect username or password.")
    return False

# Main Streamlit app function
def main():
    """Main function to run the Streamlit app."""
    st.title("Intrusion Detection System")

    # Sidebar for user authentication
    st.sidebar.title("User Authentication")
    action = st.sidebar.selectbox("Choose an option", ["Login", "Register"])
    username = st.sidebar.text_input("Username:")
    password = st.sidebar.text_input("Password:", type='password')

    if st.sidebar.button(action):
        if action == "Register":
            register_user(username, password)
        else:
            if login_user(username, password):
                st.session_state.logged_in = True

    # Main functionality after login
    if 'logged_in' in st.session_state and st.session_state.logged_in:
        uploaded_file = st.file_uploader("Upload Dataset (CSV)", type='csv')
        if uploaded_file is not None:
            data = load_data(uploaded_file)
            if data is not None:
                st.write("Data Preview:", data.head())

                model_name = st.selectbox("Select Model", ["Random Forest", "Decision Tree", "SVM"])
                if st.button("Train Model"):
                    if data.shape[1] < 2:
                        st.error("Dataset must contain at least two columns (features and target).")
                        return
                    
                    X = data.iloc[:, :-1]  # Features
                    y = data.iloc[:, -1]   # Target

                    if y.nunique() < 2:
                        st.error("Target variable must have at least two classes.")
                        return

                    model, accuracy, report, predictions, probabilities = train_model(X, y, model_name)
                    
                    if model:
                        save_model(model, 'model.joblib')
                        st.write(f"Model Accuracy: {accuracy:.2f}")
                        st.write("Classification Report:", report)

                        plot_confusion_matrix(y, predictions, model_name)
                        plot_roc_curve(y, probabilities, model_name)

                        st.image(f'confusion_matrix_{model_name}.png', caption='Confusion Matrix')
                        st.image(f'roc_curve_{model_name}.png', caption='ROC Curve')

if __name__ == "__main__":
    main()
