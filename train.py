import os
import urllib.request
import zipfile
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, confusion_matrix
import joblib

DATA_FILE = "spam.csv"
MODEL_FILE = "model.joblib"
VECTORIZER_FILE = "vectorizer.joblib"

def clean_text(text):
    """Clean the SMS text: lowercase, remove special characters and extra spaces."""
    text = str(text).lower()
    # Remove all non-word characters and numbers (optional, but good for pure text models)
    text = re.sub(r'[^a-z\s]', '', text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def train_and_compare_models():
    print("Loading data from Kaggle dataset...")
    if not os.path.exists(DATA_FILE):
        print(f"Error: {DATA_FILE} not found. Please download it from Kaggle and place it in this folder.")
        return
        
    df = pd.read_csv(DATA_FILE, encoding='latin-1')
    
    # The Kaggle dataset uses 'v1' for label and 'v2' for the message
    df = df[['v1', 'v2']]
    df.columns = ['label', 'message']
    
    print(f"Original Data shape: {df.shape}")
    
    # 1. Clean data
    print("Cleaning data...")
    df['clean_message'] = df['message'].apply(clean_text)
    
    X = df['clean_message']
    y = df['label']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 2. Vectorize using TF-IDF
    print("Vectorizing data...")
    vectorizer = TfidfVectorizer(stop_words='english', max_df=0.95)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    # 3. Train models
    print("\n--- Training Naive Bayes ---")
    nb_model = MultinomialNB()
    nb_model.fit(X_train_vec, y_train)
    nb_pred = nb_model.predict(X_test_vec)
    
    print("\n--- Training Logistic Regression ---")
    lr_model = LogisticRegression(max_iter=1000)
    lr_model.fit(X_train_vec, y_train)
    lr_pred = lr_model.predict(X_test_vec)
    
    # 4. Evaluate models
    def evaluate(name, y_true, y_pred):
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, pos_label='spam')
        rec = recall_score(y_true, y_pred, pos_label='spam')
        cm = confusion_matrix(y_true, y_pred, labels=['ham', 'spam'])
        print(f"\n{name} Results:")
        print(f"Accuracy:  {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall:    {rec:.4f}")
        print(f"Confusion Matrix:\n{cm}")
        return acc

    nb_acc = evaluate("Naive Bayes", y_test, nb_pred)
    lr_acc = evaluate("Logistic Regression", y_test, lr_pred)
    
    # 5. Pick the better one and save
    if lr_acc > nb_acc:
        best_model = lr_model
        best_name = "Logistic Regression"
    else:
        best_model = nb_model
        best_name = "Naive Bayes"
        
    print(f"\nSelecting {best_name} as the best model.")
    
    print("Saving best model and vectorizer...")
    joblib.dump(best_model, MODEL_FILE)
    joblib.dump(vectorizer, VECTORIZER_FILE)
    print("Saved successfully!")

if __name__ == "__main__":
    train_and_compare_models()
