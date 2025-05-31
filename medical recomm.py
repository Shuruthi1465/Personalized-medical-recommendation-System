import streamlit as st
import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import os

if os.path.exists("medical_condition_model.h5"):
    model = load_model("medical_condition_model.h5")
else:
    raise FileNotFoundError("Model file not found. Please train and save the model.")
    
model.save("medical_condition_model.h5")

# Load model and assets
model = load_model("medical_condition_model.h5")
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)
with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# Load dataset
df = pd.read_excel(r"D:\medical data.xlsx").dropna()

# Vectorize symptoms for content-based filtering
vectorizer = TfidfVectorizer()
symptom_vectors = vectorizer.fit_transform(df['Symptoms'].astype(str))

# Streamlit UI
st.set_page_config(page_title="Medical Recommender", layout="centered")
st.title("🩺 Personalized Medical Recommendation System")
st.markdown("""
This system predicts your **medical condition** and recommends:
- Drug name
- Drug Class
- Dosage
- Side Effects

Supports **symptom-based** or **condition-based** queries.
""")

# Tabs for different input types
tab1, tab2 = st.tabs(["🧾 Symptoms Input", "📋 Condition Input"])

with tab1:
    user_symptom = st.text_area("Enter your symptoms:", placeholder="E.g. fever, headache, nausea")
    if st.button("Get Recommendation", key="symptom"):
        if not user_symptom.strip():
            st.warning("Please enter some symptoms.")
        else:
            try:
                # Predict condition
                seq = tokenizer.texts_to_sequences([user_symptom])
                padded = pad_sequences(seq, maxlen=model.input_shape[1], padding='post')
                pred = model.predict(padded)
                condition = label_encoder.inverse_transform([pred.argmax()])[0]

                # Find most similar record
                user_vec = vectorizer.transform([user_symptom])
                sim_scores = cosine_similarity(user_vec, symptom_vectors).flatten()
                top_index = np.argmax(sim_scores)
                row = df.iloc[top_index]

                # Show result
                st.success(f"**Predicted Condition**: {condition}")
                st.markdown(f"**Recommended Drug**: {row.get('Drug name', 'N/A')}")
                st.markdown(f"**Drug Class**: {row.get('Drug Class', 'N/A')}")
                st.markdown(f"**Dosage**: {row.get('Dosage', 'N/A')}")
                st.markdown(f"**Side Effects**: {row.get('Side Effects', 'N/A')}")

            except Exception as e:
                st.error(f"Error during prediction: {e}")

with tab2:
    user_condition = st.text_input("Enter known condition (optional):", placeholder="e.g. Hypertension")
    if st.button("Get Condition Info", key="condition"):
        matched = df[df['Condition'].str.lower() == user_condition.strip().lower()]
        if not matched.empty:
            row = matched.iloc[0]
            st.success(f"**Condition**: {user_condition}")
            st.markdown(f"**Symptoms**: {row.get('Symptoms', 'N/A')}")
            st.markdown(f"**Recommended Drug**: {row.get('Drug name', 'N/A')}")
            st.markdown(f"**Drug Class**: {row.get('Drug Class', 'N/A')}")
            st.markdown(f"**Dosage**: {row.get('Dosage', 'N/A')}")
            st.markdown(f"**Side Effects**: {row.get('Side Effects', 'N/A')}")
        else:
            st.warning("No matching condition found in database.")
