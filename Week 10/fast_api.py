from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from main import predict_sentiment

app = FastAPI()

# Định nghĩa dữ liệu yêu cầu (request body)
class TextInput(BaseModel):
    text: str

@app.post("/predict/")
async def predict_sentiment_api(input: TextInput):
    try:
        sentiment = predict_sentiment(input.text)
        return {"sentiment": sentiment}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Chạy FastAPI server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
