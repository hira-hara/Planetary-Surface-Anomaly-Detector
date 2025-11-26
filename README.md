# Mars Surface Anomaly Detector 🪐

A machine learning project that classifies Martian terrain features using a fine-tuned ResNet18 model. This project was built for the MAIS 202 Bootcamp (Fall 2025).

## Project Goal

To assist Mars Rover navigation by automatically classifying terrain types from raw image feeds, helping to identify potential obstacles or scientific targets.

## Tech Stack

- Model: PyTorch, ResNet18 (Transfer Learning)

- Web App: Flask (Python), HTML/CSS

- Dataset: [NASA Curiosity Rover Data Collection Set](https://www.kaggle.com/datasets/harshitstark/nasas-curiosity-rover-data-collection)

## To Run Locally

Clone the repository:

```
git clone <your-repo-link>
cd <your-repo-name>
```

## Install Dependencies:

`pip install -r requirements.txt`

## Run the App:

`python app.py`

## View in Browser:

Open http://127.0.0.1:5000 in your web browser.

## Performance

Training Accuracy: ~76%

Validation Accuracy: ~66%

The model performs best on Dunes (100% Precision) but struggles to differentiate between Loose Rock and Sedimentary Rock due to visual similarity.

## Arhitecture

app.py: Main Flask application.

mars_model.pth: Trained PyTorch model weights.

templates/: HTML frontend files.

## Notes

Might not work for Mac users since port is 5000, if so please change to 5001
