# Named Entity Recognition API (BERT + FastAPI)

A simple Named Entity Recognition (NER) system built using a fine-tuned BERT model and deployed with FastAPI. The app exposes a RESTful API and is hosted on Render.

🔗 **Live Demo:** [API Docs](https://nlpnamedentityrecognition-render-1.onrender.com/docs)

---

## Features

- Named Entity Recognition using Hugging Face Transformers
- FastAPI backend with async processing
- Token grouping logic using BIO tagging
- CORS-enabled for easy integration
- Deployed on Render cloud

---

## Model

Model used: [Mhammad2023/bert-finetuned-ner-torch](https://huggingface.co/Mhammad2023/bert-finetuned-ner-torch)

---

## 🗂️ Project Structure

nlpNamedEntityRecognition_Render/
├── app/
│ ├── main.py # FastAPI app
│ ├── utils.py # Token grouping logic
│ └── schemas.py # Pydantic request/response models
├── start.sh # Startup script for Render
├── requirements.txt # Python dependencies
└── README.md

---

## Setup & Run Locally

### 1. Clone the Repository
```bash
git clone https://github.com/Mhammad2022Ibrahim/nlpNamedEntityRecognition_Render
cd nlpNamedEntityRecognition_Render
```
### 2. Install Dependencies
```bash
pip install -r requirements.txt
```
3. Run the FastAPI App
```bash
uvicorn app.main:app --host 0.0.0.0 --port 10000
```
---

## API Usage
### POST /token_classification
#### Request:

```json
{
  "text": "I am Mhammad Ibrahim and I work in Eurisko at Beirut."
}
```
#### Response:

```json
[
  {
    "entity_group": "PER",
    "score": 0.9873,
    "word": "Mhammad Ibrahim",
    "start": 5,
    "end": 21
  },
  ...
]
```
#### Try it live: [Swagger Docs](https://swagger.io/docs/)

---

## Deployment (Render)
### start.sh
```bash
#!/bin/bash
uvicorn app.main:app --host=0.0.0.0 --port=10000
```
### Render Settings
#### Build Command: 
```bash
pip install -r requirements.txt
```

#### Start Command: 
```bash
start.sh
```
---
