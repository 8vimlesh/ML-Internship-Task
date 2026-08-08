import streamlit as st
import joblib
import re
import os

st.set_page_config(page_title="Spam Detector", page_icon="🛡️")

st.title("🛡️ SMS Spam Detector")
st.write("Type a message below to instantly detect if it's Spam or Ham!")

# Load model and vectorizer
@st.cache_resource
def load_components():
    if os.path.exists("model.joblib") and os.path.exists("vectorizer.joblib"):
        model = joblib.load("model.joblib")
        vectorizer = joblib.load("vectorizer.joblib")
        return model, vectorizer
    else:
        return None, None

model, vectorizer = load_components()

if model is None or vectorizer is None:
    st.error("Model not found! Please run `python train.py` first to train the model.")
else:
    user_input = st.text_area("Enter your message:", height=150)
    
    if st.button("Analyze Message"):
        if not user_input.strip():
            st.warning("Please enter a message to analyze.")
        else:
            # Clean the text identical to training
            cleaned_message = user_input.lower()
            cleaned_message = re.sub(r'[^a-z\s]', '', cleaned_message)
            cleaned_message = re.sub(r'\s+', ' ', cleaned_message).strip()
            
            # Vectorize and Predict
            vec_message = vectorizer.transform([cleaned_message])
            prediction = model.predict(vec_message)[0]
            proba = model.predict_proba(vec_message)[0]
            confidence = max(proba)
            
            st.markdown("### Result")
            if prediction == 'spam':
                st.error(f"**🚨 SPAM DETECTED** (Confidence: {confidence*100:.1f}%)")
            else:
                st.success(f"**✅ SAFE MESSAGE (HAM)** (Confidence: {confidence*100:.1f}%)")
