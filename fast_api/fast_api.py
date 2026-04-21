import numpy as np
import joblib
from fastapi import FastAPI
from pydantic import BaseModel

encoder = joblib.load("fast_api/encoder.pkl")
model = joblib.load("fast_api/regr.pkl")

class PredictValue(BaseModel):
    YearRemodAdd: int
    GrLivArea: float
    TotRmsAbvGrd: int
    GarageArea: float
    TotalBsmtSF: float
    OverallQual: int
    OverallCond: int
    GarageCars: int
    MSZoning: str
    Neighborhood: str

app = FastAPI()
@app.get("/")
async def root():
    return {"message": "Hello"}
@app.post("/predict")
async def predict(request: PredictValue):
    numeric_features = [[request.YearRemodAdd, request.GrLivArea, request.TotRmsAbvGrd,
    request.GarageArea, request.TotalBsmtSF, request.OverallQual, request.OverallCond, request.GarageCars]]
    str_features = [[request.MSZoning, request.Neighborhood]]
    str_encode_features = encoder.transform(str_features)
    features = np.hstack([numeric_features, str_encode_features])
    predict = model.predict(features)
    return {"predicted score": int(predict[0])}