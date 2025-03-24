# AI-Powered Harm Detection System

## 🚀 Overview
This project is an **AI-powered NLP system** that detects harmful content in **text, URLs, and QR codes**. It uses **Machine Learning (XGBoost) with TF-IDF vectorization** to classify inputs into categories: **Hate Speech, Offensive, or Neutral**.

## 🔥 Features
- **🧠 AI-Based Harm Detection** - Uses ML to classify harmful content.
- **🔗 URL Analysis** - Extracts and analyzes webpage text.
- **📷 QR Code Scanner** - Extracts links from QR codes and checks them.
- **📊 Analytics Dashboard** - Tracks history, visualizations, and trends.
- **📄 PDF Report Generator** - Exports a session report.
- **🔐 Secure Login** - User-based tracking.

## 🛠 Tech Stack
- **Frontend**: Streamlit
- **Machine Learning**: XGBoost (Trained with TF-IDF)
- **Backend**: Python
- **Libraries**:
  - `scikit-learn`, `xgboost`: Model training & evaluation
  - `pickle`: Model storage
  - `requests`, `BeautifulSoup`: Web scraping
  - `opencv-python`, `pyzbar`: QR code scanning
  - `matplotlib`, `wordcloud`: Data visualization
  - `fpdf`: PDF generation

## 📁 Repository Structure
```
AI-Harm-Detection/
│── models/               # Trained ML models (model.pkl, vectorizer.pkl)
│── data/                 # Dataset samples (train/test)
│── notebooks/            # Jupyter notebooks for training & evaluation
│── scripts/              # Preprocessing & training scripts
│── app.py                # Main Streamlit app
│── requirements.txt       # Dependencies
│── README.md              # Documentation
│── config.yaml            # Configuration settings
│── docs/                  # Additional documentation
│── tests/                 # Unit tests for API and ML model
```

## 🎯 Dataset & Model Training
- **Dataset**: Hate Speech and Offensive Language Dataset (Kaggle)
- **Preprocessing**:
  - Text Cleaning (Removing stopwords, punctuation, lowercasing)
  - TF-IDF Vectorization
- **Model Used**: XGBoost (Optimized for high accuracy)


## ⚡ Installation
### Prerequisites
Ensure you have **Python 3.8+** installed.

### Step 1: Clone the Repository
```bash
git clone https://github.com/Krupa-Sai-S/AI-Harm-Detection.git
cd AI-Harm-Detection
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Run the Application
```bash
streamlit run app.py
```

## 🎯 How to Use
1. **Login** with your credentials.
2. **Select an action:**
   - **Detector**: Analyze text for harmful content.
   - **URL Check**: Enter a URL and analyze its content.
   - **QR Code Scanner**: Upload a QR image and check its embedded link.
   - **Analytics**: View label distribution and trends.
   - **Report**: Download a PDF of your session history.
3. **Logout** when finished.



## 👨‍💻 Contributors
- **Krupa Sai Sammangi**
- **Karthik pilli**

## 📜 License
This project is licensed under the MIT License.

## 📩 Contact
For inquiries, reach out to **Krupa Sai Sammangi**.

---
### Notes:
- Ensure `model.pkl` and `vectorizer.pkl` are available in `models/`.
- If using a different ML model, update `load_model()` accordingly.
- Future improvements: Upgrade model to **BERT** for even higher accuracy.

