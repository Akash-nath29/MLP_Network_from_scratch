from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import numpy as np
from PIL import Image
import json
from neuralnet.model import MLPNetwork

app = FastAPI()

# Load the pre-trained model
def load_model(model_path: str, input_size: int, hidden_size: int, output_size: int):
    with open(model_path, 'r') as f:
        model_data = json.load(f)
    
    nn = MLPNetwork(input_size, hidden_size, output_size)
    nn.W1 = np.array(model_data['W1'])
    nn.b1 = np.array(model_data['b1'])
    nn.W2 = np.array(model_data['W2'])
    nn.b2 = np.array(model_data['b2'])
    
    return nn

model = load_model('model.json', input_size=784, hidden_size=64, output_size=10)

def preprocess_image(image: Image.Image):
    image = image.convert('L')
    image = image.resize((28, 28))
    image_array = np.array(image) / 255.0
    image_array = image_array.flatten().reshape(1, -1)
    return image_array

@app.post("/predict/")
async def predict_digit(file: UploadFile = File(...)):
    try:
        
        image = Image.open(file.file)
        
        
        preprocessed_image = preprocess_image(image)
        
        
        prediction = model.predict(preprocessed_image)
        predicted_digit = int(prediction[0])
        
        return JSONResponse(content={"digit": predicted_digit})
    
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)