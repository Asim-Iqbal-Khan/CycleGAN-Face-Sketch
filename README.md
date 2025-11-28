# Face-Sketch-CycleGAN

A TensorFlow implementation of CycleGAN for bidirectional image-to-image translation between **Human Faces** and **Face Sketches**. This project includes a Flask web application that automatically detects the input type (Photo or Sketch) and translates it to the opposite domain.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Flask](https://img.shields.io/badge/Flask-Web%20App-green)

## 📝 Description

This project utilizes Generative Adversarial Networks (GANs), specifically the **CycleGAN** architecture, to perform unpaired image-to-image translation. Unlike standard GANs that require paired examples (e.g., a specific photo mapped to its specific sketch), CycleGAN learns to translate between domains using unpaired datasets.

The repository features a trained model capable of:
1.  **Photo $\rightarrow$ Sketch:** Converting real human faces into artistic sketches.
2.  **Sketch $\rightarrow$ Photo:** reconstructing realistic faces from line drawings.

## ✨ Features

* **CycleGAN Architecture:** Implements two Generators (G, F) and two Discriminators (X, Y) with cycle-consistency loss.
* **Bidirectional Translation:** A single web interface handles translations in both directions.
* **Auto-Detection:** The Flask app analyzes image saturation to automatically determine if the uploaded file is a Sketch or a Photo.
* **Web Interface:** A simple, user-friendly HTML/Flask frontend for uploading and viewing results.

## 📂 Dataset

The model is trained on the **Person Face Sketches** dataset from Kaggle.
* **Source:** [Kaggle - Person Face Sketches](https://www.kaggle.com/datasets/almightyj/person-face-sketches)
* **Size:** ~4,000 images (Photos and Sketches).

## 🛠️ Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/Face-Sketch-CycleGAN.git](https://github.com/your-username/Face-Sketch-CycleGAN.git)
    cd Face-Sketch-CycleGAN
    ```

2.  **Install dependencies:**
    ```bash
    pip install tensorflow flask opencv-python numpy matplotlib opendatasets
    ```

## 🚀 Usage

### 1. Training the Model
To train the model from scratch, open `Face_Sketch_CycleGAN_Training.ipynb` (formerly `i220787_q1_(1).ipynb`) in Google Colab or a local Jupyter environment.

* **Data Download:** The notebook automatically downloads the dataset using `opendatasets`. You will need your Kaggle API credentials.
* **Checkpoints:** The code is configured to save checkpoints to Google Drive to prevent data loss during long training sessions.

### 2. Running the Web Application
Once you have trained the models, save the generators as `photo_to_sketch_generator.h5` and `sketch_to_photo_generator.h5` in the root directory.

Run the Flask app:
```bash
python app.py
