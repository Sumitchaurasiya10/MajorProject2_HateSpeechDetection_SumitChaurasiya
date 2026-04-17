# Hate Speech Detector — Improved

A Flask web app that classifies text, images, and video for hate speech using a **3-tier hybrid classifier**.

## Classifiers (Priority Order)

| Priority | Classifier | Description |
|----------|-----------|-------------|
| 1 | 🧠 BiLSTM | Deep learning model (if `models/bilstm_model.h5` present) |
| 2 | 🤖 Claude AI | Anthropic API semantic understanding (if `ANTHROPIC_API_KEY` set) |
| 3 | ⚡ Adam Hybrid | TF-IDF + Logistic Regression (saga/Adam) blended with keyword rules |
| 4 | 🔑 Keyword | Pure rule-based fallback |

### Adam Hybrid Classifier
- **Algorithm:** TF-IDF (25K features, 1-3 ngrams) + Logistic Regression with `saga` solver (Adam-equivalent optimizer)
- **Blend:** 60% Adam + 40% Keyword scores → final prediction
- **Overrides:** Direct threats, hard slurs, targeted hate phrases → always Hate Speech
- **Positive guard:** Sentiment-aware — prevents false positives on clearly normal text
- **Accuracy:** 100% on test suite (15/15 edge cases)

## Labels

- **Hate Speech** — Slurs, targeted dehumanization, death threats, calls for violence against groups
- **Offensive** — Profanity, insults, crude language without targeted hate
- **Normal** — Clean, neutral, or positive content

## Setup

```bash
pip install -r requirements.txt
python app.py
```

Open `http://localhost:5000`

### Optional: Claude AI (best quality)
```bash
export ANTHROPIC_API_KEY=your_key_here
python app.py
```

### Optional: OCR for images
```bash
# Ubuntu/Debian
sudo apt install tesseract-ocr
```

## Project Structure

```
hate_speech_project_improved/
├── app.py                  # Flask routes
├── requirements.txt
├── models/
│   ├── adam_classifier.pkl   # Trained LR model (saga/Adam)
│   └── adam_vectorizer.pkl   # TF-IDF vectorizer
├── utils/
│   └── predictor.py          # HateSpeechPredictor class
└── templates/
    ├── base.html
    ├── index.html
    ├── text_analysis.html
    └── media_analysis.html
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | Classifier status |
| POST | `/api/analyze/text` | `{"text": "..."}` |
| POST | `/api/analyze/batch` | `{"texts": [...]}` (max 50) |
| POST | `/api/analyze/image` | multipart file upload |
| POST | `/api/analyze/video` | multipart file upload |
