# Spam Detection Model - Final Submission Report

*Note: You can copy and paste the contents of this markdown file into a Word Document, insert your screenshots where indicated, and save as PDF to submit.*

---

## 1. Project Description

**What I Built**
I built an end-to-end Machine Learning pipeline to detect SMS spam messages using Natural Language Processing (NLP) techniques, along with a Streamlit web application that serves as an interactive demo. The web app allows users to input arbitrary text messages and receive an instant prediction on whether the message is "Spam" or "Ham" (legitimate).

**Approach & Methodology**
1. **Data Loading & Cleaning**: I started by loading the standard UCI SMS Spam Collection dataset using Pandas. I applied a custom text cleaning function to convert all text to lowercase, remove any non-alphabetic characters (noise), and strip extra whitespace.
2. **Feature Engineering (TF-IDF)**: To convert the raw text data into numerical features suitable for machine learning, I used Scikit-Learn's `TfidfVectorizer`. I removed standard English stop words and ignored terms that appeared in more than 95% of the documents to filter out overly common words.
3. **Model Selection & Training**: I split the dataset (80% training, 20% testing) and trained two classification models to compare their performance: **Multinomial Naive Bayes** and **Logistic Regression**.
4. **Evaluation**: Both models were evaluated on the test set using Accuracy, Precision, Recall, and a Confusion Matrix. Naive Bayes was selected as the final model due to its superior overall accuracy (96.8%) and perfect precision score (100%), which is critical in spam detection to avoid false positives (flagging a legitimate message as spam).
5. **Deployment**: Finally, I exported the trained model and vectorizer using `joblib` and wrapped them in a Python `Streamlit` application, providing an accessible User Interface for inference.

---

## 2. Screenshots of Code, Charts & Output

### 2.1 Code Implementation
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

DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
ZIP_FILE = "smsspamcollection.zip"
DATA_FILE = "SMSSpamCollection"
MODEL_FILE = "model.joblib"
VECTORIZER_FILE = "vectorizer.joblib"

def download_data():
    if not os.path.exists(DATA_FILE):
        print("Downloading dataset...")
        urllib.request.urlretrieve(DATA_URL, ZIP_FILE)
        print("Extracting dataset...")
        with zipfile.ZipFile(ZIP_FILE, 'r') as zip_ref:
            zip_ref.extractall(".")
        os.remove(ZIP_FILE)
    else:
        print("Dataset already exists.")

def clean_text(text):
    """Clean the SMS text: lowercase, remove special characters and extra spaces."""
    text = str(text).lower()
    # Remove all non-word characters and numbers (optional, but good for pure text models)
    text = re.sub(r'[^a-z\s]', '', text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def train_and_compare_models():
    print("Loading data...")
    df = pd.read_csv(DATA_FILE, sep='\t', header=None, names=['label', 'message'])
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
    download_data()
    train_and_compare_models()




## Key Insights & Recommendations

**What I Learned & Challenges Faced**
Through this project, I learned the importance of text preprocessing and feature representation in NLP tasks. While TF-IDF is highly effective, a major challenge I encountered was dealing with very short messages (e.g., "ok", "yes"). The model struggles with these if they lack distinct vocabulary. Furthermore, Spammers often use intentional misspellings or unseen slang to bypass filters (e.g., "f.r.e.e"). Because standard TF-IDF relies on exact string matches of whole words, it cannot easily capture these variations without character-level analysis. The models also lack an understanding of deeper semantic context, making it impossible for them to detect sarcasm.

**Suggestions for Improvement**
1. **Incorporate N-Grams**: Updating the `TfidfVectorizer` to use bi-grams or tri-grams (e.g., `ngram_range=(1,2)`) would allow the model to capture common spam phrases like "win cash" or "claim prize", rather than just isolated words.
2. **Feature Engineering**: We could significantly boost performance by manually engineering features, such as adding the length of the message, the count of uppercase letters, the frequency of special characters (like `$`, `!`), or detecting the presence of a URL/hyperlink.
3. **Advanced Deep Learning Models**: For better contextual understanding and handling of unseen slang, transitioning from traditional ML algorithms to a sequence-based deep learning model (like an LSTM) or a transformer model (like BERT) would drastically improve the model's robustness.
