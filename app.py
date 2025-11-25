import os
import torch
import torch.nn as nn
from flask import Flask, request, render_template, redirect
from torchvision import transforms, models
from PIL import Image

app = Flask(__name__)

# --- CONFIGURATION ---
UPLOAD_FOLDER = 'static/uploads'
MODEL_PATH = 'mars_model.pth'
# MUST match the exact order of classes from your Colab training
CLASSES = ['Altered Rock', 'Bedrock', 'Dunes', 'Loose Rock', 'Sedimentary Rock']

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- LOAD MODEL ---
def load_mars_model():
    try:
        # 1. Recreate the architecture (ResNet18)
        model = models.resnet18(pretrained=False)
        num_ftrs = model.fc.in_features
        # 2. Reconfigure the final layer to 5 classes
        model.fc = nn.Linear(num_ftrs, len(CLASSES))
        
        # 3. Load the weights (Force loading to CPU)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
        model.eval()
        print("✅ Model loaded successfully!")
        return model
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None

model = load_mars_model()

# --- PREPROCESSING ---
def process_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0) # Add batch dimension

# --- ROUTES ---
@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    confidence = None
    img_url = None

    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        
        file = request.files['file']
        
        if file.filename == '':
            return redirect(request.url)

        if file:
            # Save the file temporarily
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            img_url = filepath

            # Predict
            if model:
                try:
                    tensor = process_image(filepath)
                    outputs = model(tensor)
                    
                    # Get probabilities
                    probabilities = torch.nn.functional.softmax(outputs, dim=1)
                    top_prob, top_class = torch.max(probabilities, 1)
                    
                    prediction = CLASSES[top_class.item()]
                    confidence = f"{top_prob.item() * 100:.1f}%"
                except Exception as e:
                    print(f"Prediction Error: {e}")
                    prediction = "Error processing image"

    return render_template('index.html', prediction=prediction, confidence=confidence, img_url=img_url)

if __name__ == '__main__':
    # Run the app locally on port 5000
    app.run(debug=True, port=5000)
