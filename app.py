from starlette.requests import Request
from starlette.middleware.cors import CORSMiddleware
from PIL import Image
from http.client import HTTPResponse
import io
import numpy as np
import uvicorn
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
from io import BytesIO
from typing import Tuple
from gtts import gTTS
import IPython.display as ipd
import os
from fastapi import Query
from typing import Dict

# Define a function to convert text to audio and play it
# def text_to_audio(text, output_file='prediction_audio.mp3'):
#     tts = gTTS(text=text, lang='en')
#     tts.save(output_file)
#     os.system(f"start {output_file}")
#     return output_file


# Define a dictionary to map class indices to class labels
# class_mapping = {
#     0: 'fish and chips',
#     1: 'french toast',
#     2: 'fried calamari',
#     3: 'garlic bread',
#     4: 'grilled salmon',
#     5: 'hamburger',
#     6: 'ice cream',
#     7: 'lasagna',
#     8: 'macaroni and cheese',
#     9: 'macarons'
# }

app = FastAPI()

# Allow all origins for CORS (you might want to restrict this in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the model
model = tf.keras.models.load_model('/home/fadwa/code/Ghita-kh/NFG_repo/best_model.h5')
# The class names of the model
CLASS_NAMES = ['fish_and_chips', 'french_toast', 'fried_calamari', 'garlic_bread', 'grilled_salmon', 'hamburger', 'ice_cream', 'lasagna', 'macaroni_and_cheese', 'macarons']

@app.get("/")
async def read_root():
    return {"Hello": "World"}

@app.get("/favicon.ico")
async def get_favicon():
    return HTTPResponse(content=b"", status_code=200)

@app.get("/ping")
async def ping():
    return "hello , i am alive"


def read_file_as_image(data) -> Tuple[np.ndarray, Tuple[int, int]]:
    img = Image.open(BytesIO(data)).convert('RGB')
    img_resized = img.resize((224, 224), resample=Image.BICUBIC)
    image = np.array(img_resized)
    return image, img_resized.size


# # Define the API endpoint for image classification
# @app.post("/classify-image")
# async def classify_image(request: Request, file: UploadFile = File(...)):
#     contents = await file.read()
#     image = Image.open(io.BytesIO(contents)).resize((224, 224))

#     predictions = predict_image(image)

#     return JSONResponse(content={"predictions": predictions})



@app.post("/predict")
async def predict(file: UploadFile = File(...)): # The function that will be executed when the endpoint is called
    try: # A try block to handle any errors that may occur
        image, img_size = read_file_as_image(await file.read()) # Read the image file
        img_batch = np.expand_dims(image, 0) # Add an extra dimension to the image so that it matches the input shape of the model

        predictions = model.predict(img_batch) # Make a prediction
        predicted_class = CLASS_NAMES[np.argmax(predictions[0])] # Get the predicted class
        confidence = np.max(predictions[0]) # Get the confidence of the prediction

        return { # Return the prediction
            'class': predicted_class,
            'confidence': float(confidence)
        }
    except Exception as e: # If an error occurs
        raise HTTPException(status_code=400, detail=str(e)) # Raise an HTTPException with the error message

# @app.get("/predict")
# async def predict(image_path: str) -> Dict[str, float]:
#     try:
#         image, img_size = read_file_as_image(image_path)
#         img_batch = np.expand_dims(image, 0)

#         predictions = model.predict(img_batch)
#         predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
#         confidence = np.max(predictions[0])
#         predicted_class_index = np.argmax(predictions[0])
#         # Predicted class index
#         #predicted_class_index = predicted_class_index  # Replace with the actual predicted class index

# # Get the predicted class label from the mapping
#         # predicted_label = class_mapping[predicted_class_index]

#         # Convert the predicted class label to audio and play it
#         # audio_file = text_to_audio(predicted_label)
#         # ipd.Audio(audio_file)
#         return {
#             'class': predicted_class,
#             'confidence': float(confidence),
#             #  'audio' : ipd.Audio(audio_file)
#         }
#     except Exception as e:
#         raise HTTPException(status_code=400, detail=str(e))

@app.get("/", response_class=HTMLResponse)
async def index():
    with open("main.html", "r") as file:
        return file.read()


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
