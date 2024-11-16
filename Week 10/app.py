# app.py
import streamlit as st
import requests

st.title("Sentiment Analysis with PhoBERT")
st.title("240106 - Nguyen Luu Phuong Ngoc Lam - Week 10")

st.markdown("### Nhập văn bản để phân tích cảm xúc:")

# User input
text_input = st.text_area("Văn bản đầu vào:", height=200)

if st.button("Dự đoán cảm xúc"):
    if text_input.strip():
        # Gửi yêu cầu đến FastAPI
        response = requests.post("http://127.0.0.1:8000/predict/", json={"text": text_input})
        
        if response.status_code == 200:
            sentiment = response.json().get("sentiment", "Không thể dự đoán cảm xúc")
            st.write(f"**Cảm xúc dự đoán:** {sentiment}")
        else:
            st.write("Có lỗi xảy ra khi xử lý yêu cầu.")
    else:
        st.write("Hãy nhập một đoạn văn bản hợp lệ!")
