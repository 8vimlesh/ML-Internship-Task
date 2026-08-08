# SMS Spam Detector

This project implements a Machine Learning pipeline and a Streamlit web application to classify SMS messages as either **Spam** or **Ham** (legitimate).

## Project Overview
1. **Data Preprocessing**: Cleans raw text data (removes noise, converts to lowercase) and transforms it into numerical features using `TfidfVectorizer`.
2. **Modeling**: Trains and compares a `MultinomialNB` (Naive Bayes) and `LogisticRegression` model. 
3. **Evaluation**: Models are evaluated using Accuracy, Precision, Recall, and Confusion Matrices. Naive Bayes was automatically selected as the final model due to its high accuracy (96.8%) and perfect 100% precision score on the test set.
4. **Web Demo**: A Streamlit application provides an interactive UI to instantly test new SMS messages.

## Dataset
The model is trained on the standard SMS Spam Collection dataset.
- Download the dataset from Kaggle: [SMS Spam Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)
- Extract `spam.csv` and place it in the root directory of this project before training.

## Setup & Installation
1. Clone this repository to your local machine.
2. Install the required Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
### 1. Training the Model
To re-train the model and generate the evaluation charts (requires `spam.csv`), run:
```bash
python train.py
python generate_charts.py
```
This will output `model.joblib` and `vectorizer.joblib`.

### 2. Running the Web App
To start the interactive Streamlit dashboard, run:
```bash
streamlit run app.py
```
Then, open your web browser to `http://localhost:8501` to use the app!
