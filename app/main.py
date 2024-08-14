from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
import json
import pandas as pd
import database
from sklearn.preprocessing import LabelEncoder
import time
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from sqlalchemy.orm import Session
from sqlalchemy import func
from .database import SessionLocal, RequestLog

app = FastAPI()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Import the data
with open('/app/app/idmanual.json', 'r') as file:
    data = json.load(file)

# Convert the JSON data to a pandas DataFrame
df = pd.DataFrame(data)
num_labels = len(df['class_id'].unique())

# Load the model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_labels)
model.load_state_dict(torch.load("/app/app/trademark_classification_model.pth"))
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)
model.eval()

# Load the tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

label_encoder = LabelEncoder()
df['class_id_encoded'] = label_encoder.fit_transform(df['class_id'])
df['status_encoded'] = df['status'].map({'A': 1, 'D': 0})

class PredictRequest(BaseModel):
    description: str
    user_id: str

class PredictResponse(BaseModel):
    predicted_class: str
    inference_time: float

@app.post("/predict/", response_model=PredictResponse)
async def predict(request: PredictRequest, db: Session = Depends(get_db)):
    # Count the number of requests made by the user
    request_count = db.query(func.count(RequestLog.id)).filter(RequestLog.user_id == request.user_id).scalar()

    # Check if the user has exceeded 5 requests
    if request_count >= 5:  # It allows only 5 and request_count starts from 0
        raise HTTPException(status_code=429, detail="Too many requests")

    # Tokenize the input description
    inputs = tokenizer(request.description, return_tensors="pt", padding=True, truncation=True).to(device)

    # Perform inference
    start_time = time.time()
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        prediction = torch.argmax(logits, dim=1).item()
    end_time = time.time()

    # Convert prediction back to class label
    predicted_class = label_encoder.inverse_transform([prediction])[0]
    inference_time = end_time - start_time

    # Store request in the database
    db.add(RequestLog(user_id=request.user_id, description=request.description, inference_time=inference_time))
    db.commit()

    # Return the prediction and inference time
    return PredictResponse(predicted_class=predicted_class, inference_time=inference_time)
