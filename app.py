import streamlit as st
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from datetime import datetime
from fpdf import FPDF
import io
import requests
from bs4 import BeautifulSoup
import cv2
import numpy as np
import tempfile

# -------------------- PAGE SETUP -------------------- #
st.set_page_config(page_title="AI-Powered-Harm-Detection-System", layout="wide")

# -------------------- USERS & THEME -------------------- #
users = {
    "krupa sai": "1234",
    "judge": "hackathon",
    "admin": "admin123",
    "karthik pilli": "1432"
}

if "theme" not in st.session_state:
    st.session_state.theme = "Light"

# -------------------- LOGIN -------------------- #
if "username" not in st.session_state:
    with st.form("login_form"):
        st.title("🔐 Secure Login")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")

        if submitted:
            if username in users and users[username] == password:
                st.session_state.username = username
                st.session_state.view = "Detector"
                st.success(f"✅ Welcome back, {username}!")
                st.rerun()
            else:
                st.error("❌ Invalid username or password.")

# -------------------- LOGGED-IN VIEWS -------------------- #
if "username" in st.session_state:
    st.sidebar.title(f"👤 {st.session_state.username}")
    view = st.sidebar.radio("Navigation", ["Detector", "URL Check", "Analytics", "Report", "About"], index=0)
    st.session_state.view = view

    st.sidebar.markdown("---")
    if st.sidebar.button("🚪 Logout"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

    # -------------------- LOAD MODEL -------------------- #
    @st.cache_resource
    def load_model():
        with open("model.pkl", "rb") as f:
            model = pickle.load(f)
        with open("vectorizer.pkl", "rb") as f:
            vectorizer = pickle.load(f)
        return model, vectorizer

    model, vectorizer = load_model()

    label_map = {
        0: "Hate Speech",
        1: "Offensive",
        2: "Neutral"
    }

    def classify(text):
        vector = vectorizer.transform([text])
        prediction = model.predict(vector)[0]
        return label_map[prediction]

    if "history" not in st.session_state:
        st.session_state.history = []

    if "trigger_classify" not in st.session_state:
        st.session_state.trigger_classify = False

    # -------------------- DETECTOR -------------------- #
    if st.session_state.view == "Detector":
        st.title("🧠 AI Powered Harm Detection System")
        example = st.selectbox("Try an example:", ["", "You are horrible", "Thanks for your help!", "I hate this."])
        text_input = st.text_area("Enter text to classify", value=example)

        if st.button("Classify"):
            st.session_state.trigger_classify = True

        if st.session_state.trigger_classify:
            if text_input.strip():
                result = classify(text_input)
                st.success(f"Prediction: **{result}**")

                st.session_state.history.append({
                    "name": st.session_state.username,
                    "text": text_input,
                    "label": result,
                    "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
            else:
                st.warning("Please enter valid text.")
            st.session_state.trigger_classify = False

    # -------------------- URL HARM DETECTION -------------------- #
    elif st.session_state.view == "URL Check":
        st.title("🔗 URL Harm Detection")
        
        url_input = st.text_input("Enter a URL to analyze")
        
        def extract_text_from_url(url):
            try:
                response = requests.get(url, timeout=5)
                if response.status_code == 200:
                    soup = BeautifulSoup(response.text, "html.parser")
                    page_text = " ".join([p.text for p in soup.find_all("p")])
                    return page_text[:1000]
                else:
                    return None
            except Exception as e:
                return None

        if st.button("Analyze URL"):
            if url_input.strip():
                with st.spinner("Fetching website content..."):
                    webpage_text = extract_text_from_url(url_input)
                
                if webpage_text:
                    result = classify(webpage_text)
                    st.success(f"Prediction: **{result}**")
                else:
                    st.error("Failed to fetch content from the URL. It might be blocked or invalid.")
            else:
                st.warning("Please enter a valid URL.")

    # -------------------- ANALYTICS -------------------- #
    elif st.session_state.view == "Analytics":
        st.title("📊 Prediction Analytics")
        df = pd.DataFrame(st.session_state.history)
        if not df.empty:
            user_df = df[df["name"] == st.session_state.username]
            st.subheader("Your Prediction History")
            st.dataframe(user_df[::-1], use_container_width=True)

            st.subheader("Label Distribution")
            st.bar_chart(user_df["label"].value_counts())

            st.subheader("Word Cloud")
            text_blob = " ".join(user_df["text"])
            wordcloud = WordCloud(width=800, height=300, background_color="white").generate(text_blob)
            st.image(wordcloud.to_array(), use_column_width=True)
        else:
            st.info("No data to display yet.")
    # -------------------- REPORT DOWNLOAD -------------------- #
    elif st.session_state.view == "Report":
        st.title("📄 Download Session Report")

        def generate_pdf(data):
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", "B", 16)
            pdf.cell(200, 10, f"Harm Detection Report - {st.session_state.username}", ln=True, align="C")
            pdf.set_font("Arial", "", 12)
            pdf.cell(200, 10, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
            pdf.ln(10)
            for row in data[::-1]:
                pdf.multi_cell(0, 10, f"[{row['time']}] {row['text']} -> {row['label']}")
            return pdf.output(dest='S').encode('latin-1')

        user_data = [x for x in st.session_state.history if x["name"] == st.session_state.username]
        if user_data:
            pdf_bytes = generate_pdf(user_data)
            st.download_button("📥 Download PDF Report", data=pdf_bytes, file_name="harm_report.pdf")
        else:
            st.info("No predictions to report.")

    # -------------------- ABOUT -------------------- #
    elif st.session_state.view == "About":
        st.title("📘 About This App")
        st.markdown("""This AI tool detects harmful content like:
        - Hate Speech
        - Offensive Language
        - Neutral messages

        🔒 Features:
        - Secure Login
        - Text Analysis
        - URL Analysis
        - User-specific History
        - PDF Report Generator
        - Streamlit Interface
        """)
