import pickle

from fastapi import FastAPI, Request
import uvicorn
from pydantic import BaseModel

app = FastAPI(title='Lead-Prediction')

class Client(BaseModel):
    lead_source: str
    number_of_courses_viewed: int
    annual_income: float


with open('pipeline_v1.bin', 'rb') as f_in:
    pipeline = pickle.load(f_in)

def predict_single(client):
    result = pipeline.predict_proba(client)[0, 1]
    return float(result)

@app.post("/Q4-predict")
def predict(client: Client):
    prob = predict_single(client.model_dump())

    return prob

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9696)
