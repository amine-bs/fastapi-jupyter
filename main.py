from io import BytesIO
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from utils import load_device, load_model, predict, is_image_file
from PIL import Image

class Predicition(BaseModel):
    label: str
    probability: float
    description: None
 
app = FastAPI()

device = load_device()
model = load_model(device)

def read_image(file):
    img = Image.open(BytesIO(file)).convert("RGB")
    return img




@app.get("/")
def root():
    return {"message": "Welcome to Image Classification FastAPI"}
"""
@app.get("/model")
def details():
    model = "ResNet18"
    accuracy = "99.25%"
    training_dataset = "https://www.kaggle.com/datasets/rhammell/ships-in-satellite-imagery"
    return {"Model": model, 
            "accuracy": accuracy,
            "training dataset": training_dataset}
"""
@app.get("/model/{info}")
def details(info:str, n:int=2):
    accuracy = 99.2511111
    if info == 'architecture':
        return {'architecture': 'ResNet18'}
    elif info == 'dataset':
        return {'dataset url': "https://www.kaggle.com/datasets/rhammell/ships-in-satellite-imagery"}
    elif info == 'accuracy':
        formatted_accuracy = int((10**n)*accuracy)/(10**n)
        return {'accuracy': '{}%'.format(formatted_accuracy)}
    else:    
        return '{} is not available'.format(info)
    
@app.post("/predict")
async def predict_api(file: UploadFile = File(...)):
    if not is_image_file(file.filename):
        return "file must have image format"
    img = read_image(await file.read())
    preds = predict(img, model, device)
    return preds


