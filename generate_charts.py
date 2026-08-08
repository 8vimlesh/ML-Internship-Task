import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

DATA_FILE = "spam.csv"

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def generate_and_save_charts():
    print("Loading data from Kaggle dataset...")
    if not os.path.exists(DATA_FILE):
        print(f"Error: {DATA_FILE} not found.")
        return
        
    df = pd.read_csv(DATA_FILE, encoding='latin-1')
    df = df[['v1', 'v2']]
    df.columns = ['label', 'message']
    
    df['clean_message'] = df['message'].apply(clean_text)
    
    X = df['clean_message']
    y = df['label']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    vectorizer = TfidfVectorizer(stop_words='english', max_df=0.95)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    # Train Naive Bayes
    nb = MultinomialNB()
    nb.fit(X_train_vec, y_train)
    nb_pred = nb.predict(X_test_vec)
    cm_nb = confusion_matrix(y_test, nb_pred, labels=['ham', 'spam'])
    
    # Train Logistic Regression
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train_vec, y_train)
    lr_pred = lr.predict(X_test_vec)
    cm_lr = confusion_matrix(y_test, lr_pred, labels=['ham', 'spam'])
    
    sns.set_theme(style="whitegrid")
    
    # Plot Naive Bayes CM
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_nb, annot=True, fmt='d', cmap='Blues', xticklabels=['ham', 'spam'], yticklabels=['ham', 'spam'])
    plt.title("Naive Bayes Confusion Matrix")
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig('naive_bayes_cm.png', dpi=300)
    plt.close()
    print("Saved naive_bayes_cm.png")

    # Plot Logistic Regression CM
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Oranges', xticklabels=['ham', 'spam'], yticklabels=['ham', 'spam'])
    plt.title("Logistic Regression Confusion Matrix")
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig('logistic_regression_cm.png', dpi=300)
    plt.close()
    print("Saved logistic_regression_cm.png")

if __name__ == "__main__":
    generate_and_save_charts()
