from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Định nghĩa mô hình và tokenizer
MODEL_PATH = "phobert_model/phobert_model.pth"
TOKENIZER_PATH = "phobert_model/phobert_tokenizer"
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)

# Load PhoBERT model
def load_model():
    model = AutoModelForSequenceClassification.from_pretrained("vinai/phobert-base", num_labels=3)
    checkpoint = torch.load(MODEL_PATH, map_location=torch.device('cuda'))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

model = load_model()

# Nhãn cảm xúc
labels = {0: "negative", 1: "neutral", 2: "positive"}

# Hàm tiền xử lý văn bản
def preprocess_text(text):
    tokens = tokenizer.encode_plus(
        text,
        max_length=256,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )
    return tokens

# Hàm dự đoán cảm xúc
def predict_sentiment(text):
    tokens = preprocess_text(text)
    with torch.no_grad():
        outputs = model(**tokens)
    logits = outputs.logits
    prediction = torch.argmax(logits, dim=1).item()
    return labels[prediction]

# Khởi tạo FastAPI app
app = FastAPI()

class TextInput(BaseModel):
    text: str

# Route gốc
@app.get("/")
async def root():
    return {"message": "Welcome to the Sentiment Analysis API!"}

# Route dự đoán cảm xúc
@app.post("/predict/")
async def predict_sentiment_api(input: TextInput):
    try:
        sentiment = predict_sentiment(input.text)
        return {"sentiment": sentiment}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
