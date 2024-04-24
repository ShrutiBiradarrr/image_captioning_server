

from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import pickle
from PIL import Image
# from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
# from nltk.translate.bleu_score import sentence_bleu
# from tensorflow.keras.preprocessing import image as keras_image
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import base64
from io import BytesIO
import time

app = Flask(__name__)
CORS(app) 
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

# Load the trained model
new_model = tf.keras.models.load_model('model_18.h5')

# Load other necessary artifacts
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)
with open('features.pkl', 'rb') as f:
    features = pickle.load(f)

# Image preprocessing function
def preprocess_image(img):
    try:
        # Try opening image as file object
        img_pil = Image.open(img)
    except AttributeError:
        try:
            # If img is a file path, open it
            img_pil = Image.open(img.filename)
        except AttributeError:
            # If img is a string (file path), open it directly
            img_pil = Image.open(img)
    
    img_resized = img_pil.resize((224, 224))  # Adjust dimensions as needed
    img_array = keras_image.img_to_array(img_resized)
    img_preprocessed = tf.keras.applications.vgg16.preprocess_input(img_array)
    return img_preprocessed

# Function to preprocess captions
def preprocess_captions(data):
    total_captions = []

    for caption in data["caption"].astype(str):
        caption = '<start> ' + caption + ' <end>'
        total_captions.append(caption)
    return total_captions

# Load image using VGG16 preprocessing
def load_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (224, 224))
    img = tf.keras.applications.vgg16.preprocess_input(img)  # VGG16 preprocessing
    return img, image_path

# Calculate maximum and minimum lengths of captions
def calc_max_length(tensor):
    return max(len(t) for t in tensor)

def calc_min_length(tensor):
    return min(len(t) for t in tensor)
# Function to generate caption for the image
# def generate_caption(image):
#     image = preprocess_image(image)
#     input_ids = processor(image).input_ids.unsqueeze(0)
#     caption_ids = model.generate(input_ids, max_length=50, num_beams=5)
#     caption = processor.decode(caption_ids[0], skip_special_tokens=True)
#     return caption

def convert_image_to_base64(image):
    # Convert image to base64 encoding
    image_base64 = base64.b64encode(image.read()).decode('utf-8')
    return image_base64

def generate__caption(image):
    with Image.open(image) as img:
        raw_image = img.convert("RGB")

        inputs = processor(raw_image, return_tensors="pt", max_new_tokens=100)

        start_time = time.time()
        out = model.generate(**inputs)
        generation_time = time.time() - start_time

        caption = processor.decode(out[0], skip_special_tokens=True)
        return caption, generation_time

@app.route('/after', methods=['POST'])
def predict_caption():
    if 'file' not in request.files:
        return jsonify({'error': 'No image uploaded'})

    image = request.files['file']
    image_name = image.filename

    # Check if the file is empty
    if image.filename == '':
        return jsonify({'error': 'No selected file'})

    if image:
        image_base64 = convert_image_to_base64(image)
        caption, generation_time = generate__caption(image)

        response = {
            'image_name': image_name,
            'description': caption,
            'generation_time': generation_time
        }

        return jsonify(response)
    else:
        return jsonify({'error': 'Invalid file'})
    
# def predict__caption():
#     image_file = request.files['file']
    
#     # Load the image
#     image = Image.open(image_file)
    
#     # Generate caption for the image
#     real_caption = '<start> black dog is digging in the snow <end>'  # Placeholder for the real caption
    
#     start = time.time()
#     predicted_caption = generate_caption(image)
    
    # # Calculate BLEU score
    # reference = [real_caption.split()]
    # candidate = predicted_caption.split()
    # score = sentence_bleu(reference, candidate, weights=(0.5, 0.5, 0, 0))
    
    # # Prepare response
    # response = {
    #     "real_caption": real_caption,
    #     "predicted_caption": predicted_caption,
    #     "bleu_score": score * 100,
    #     "processing_time": round(time.time() - start, 2)  # Time taken for prediction
    # }
    
    # return jsonify(response)
if __name__ == '__main__':
    app.run(debug=True)
