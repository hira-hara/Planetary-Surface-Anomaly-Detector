import os
import torch
import torch.nn as nn
from flask import Flask, request, render_template, redirect, jsonify
from torchvision import transforms, models
from PIL import Image
import glob

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
MODEL_PATH = 'mars_model.pth'
# MUST match the exact order of classes from your Colab training
CLASSES = ['Altered Rock', 'Bedrock', 'Dunes', 'Loose Rock', 'Sedimentary Rock']

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load Model
def load_mars_model():
    try:
        # 1. Recreate the architecture (ResNet18)
        model = models.resnet18(weights=False)
        num_ftrs = model.fc.in_features
        # 2. Reconfigure the final layer to 5 classes
        model.fc = nn.Linear(num_ftrs, len(CLASSES))
        
        # 3. Load the weights (Force loading to CPU)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
        model.eval()
        print("Model loaded")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

model = load_mars_model()

# Preprocessing 
def process_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)), # Need image size to be adjusted for ResNet18 to work
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0) # Add batch dimension

# -- Helper --
def cleanup_folder():
    files = glob.glob(os.path.join(UPLOAD_FOLDER, '*'))
    for f in files:
        try:
            os.remove(f)
        except OSError as e:
            print(f"Failed to remove old file: {f}: {e}")


# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    global file_counter
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if file:
        # Save file
        cleanup_folder() # Cleanup first to keep stuff lightweight
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        
        if model:
            try:
                # Predict
                tensor = process_image(filepath)
                outputs = model(tensor)
                
                # Calculate Confidence
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                top_prob, top_class = torch.max(probabilities, 1)
                
                prediction = CLASSES[top_class.item()]
                confidence_score = top_prob.item() * 100
                confidence_str = f"{confidence_score:.1f}%"

                # Return JSON (Data) instead of HTML
                return jsonify({
                    'prediction': prediction,
                    'confidence': confidence_str,
                    'confidence_score': confidence_score, # Number for the bar width
                    'image_url': filepath
                })
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        else:
            return jsonify({'error': 'Model not loaded'}), 500

if __name__ == '__main__':
    # Run the app locally on port 5000
    print("App running...")
    app.run(debug=True, port=5000)

