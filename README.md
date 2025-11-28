# Face-Sketch-CycleGAN

A TensorFlow implementation of CycleGAN for bidirectional image-to-image translation between **Human Faces** and **Face Sketches**.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![License](https://img.shields.io/badge/License-MIT-green)

## 📝 Description

This project utilizes Generative Adversarial Networks (GANs), specifically the **CycleGAN** architecture, to perform unpaired image-to-image translation. Unlike standard GANs that require paired examples (e.g., a specific photo mapped to its specific sketch), CycleGAN learns to translate between domains using unpaired datasets.

The repository features a trained model capable of:
1.  **Photo $\rightarrow$ Sketch:** Converting real human faces into artistic sketches.
2.  **Sketch $\rightarrow$ Photo:** Reconstructing realistic faces from line drawings.

## ✨ Features

* **CycleGAN Architecture:** Implements two Generators (G, F) and two Discriminators (X, Y) with cycle-consistency loss.
* **Bidirectional Translation:** The model trains both directions simultaneously (Photo to Sketch and Sketch to Photo).
* **Group Normalization:** Uses GroupNormalization (acting as Instance Normalization) to preserve style independent of batch size.
* **Google Drive Integration:** The notebook includes setup for saving checkpoints directly to Google Drive to prevent data loss.

## 📂 Dataset

The model is configured to automatically download the **Person Face Sketches** dataset from Kaggle using `opendatasets`.
* **Source:** [Kaggle - Person Face Sketches](https://www.kaggle.com/datasets/almightyj/person-face-sketches)
* **Size:** ~4,000 images (Photos and Sketches).

## 📁 Project Structure

```text
├── CycleGAN_Face_Sketch.ipynb    # Main Training Notebook (Rename as needed)
├── README.md               # Project Documentation
└── results/                # Directory for comparison screenshots
    ├── photo_result.png    # (Add your images here)
    └── sketch_result.png   # (Add your images here)
```


## 🚀 Usage

1.  **Open the Notebook:**
    Open `CycleGAN_face_sketch.ipynb` in Google Colab or a Jupyter environment.

2.  **Install Dependencies:**
    Run the first cell to install `opendatasets` and import TensorFlow/Keras libraries.

3.  **Download Data:**
    Run the data loading cells. You will be prompted for your Kaggle API username and key.

4.  **Train:**
    Execute the training loop. The model uses a half-size dataset (approx 4,000 images) for efficiency.
    * **Checkpoints:** Saved to `/content/drive/MyDrive/Colab_CheckPoints/CycleGAN/train`.
    * **Final Model:** Saved locally as `photo_to_sketch_generator.h5` and `sketch_to_photo_generator.h5`.

## 📊 Results

### Photo to Sketch
| Input Photo | Generated Sketch |
| :---: | :---: |
| <img src="./results/photo_input.jpg" width="250" /> | <img src="./results/sketch_output.jpg" width="250" /> |

### Sketch to Photo
| Input Sketch | Generated Photo |
| :---: | :---: |
| <img src="./results/sketch_input.jpg" width="250" /> | <img src="./results/photo_output.jpg" width="250" /> |


## 🧠 Model Architecture

The project uses the standard CycleGAN architecture:

* **Generators:** U-Net style generators modified to use **GroupNormalization**.
* **Discriminators:** PatchGAN discriminators that classify $70 \times 70$ overlapping image patches as real or fake.
* **Loss Functions:**
    * *Adversarial Loss:* To generate realistic images.
    * *Cycle Consistency Loss:* Ensures `Photo -> Sketch -> Photo` resembles the original input.
    * *Identity Loss:* Ensures the generator doesn't change images that already belong to the target domain.

## 📜 License

This project is open-source and available under the MIT License.
