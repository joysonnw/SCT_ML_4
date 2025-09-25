# Hand Gesture Recognition (CNN)

This project builds and trains a Convolutional Neural Network (CNN) to recognize static hand gestures from the **LeapGestRecog** dataset. After training, the model can perform real-time gesture recognition using your webcam.

---

## How It Works
1. **Dataset Loading**: Images are loaded from the LeapGestRecog dataset, converted to grayscale, resized to 64×64, and normalized.  
2. **Model Architecture**:  
   - Two convolution + max pooling layers  
   - Flatten layer  
   - Dense layer with dropout for regularization  
   - Final dense layer with softmax for classification  
3. **Training**: The model is trained with categorical cross-entropy loss and Adam optimizer.  
4. **Live Demo**: Once trained, the model captures frames from your webcam, preprocesses them, and predicts the gesture in real-time.

---

## Dataset
Download the dataset here:  
[LeapGestRecog Dataset on Kaggle](https://www.kaggle.com/datasets/gti-upm/leapgestrecog)

Unzip the dataset and update the `DATASET_DIR` path in the script.

---

## Dependencies
Install the required packages before running:
```bash
pip install tensorflow opencv-python scikit-learn numpy
```

---

## How to Run
1. Clone this repository:
   ```bash
   git clone https://github.com/joysonnw/SCT_ML_4.git
   cd SCT_ML_4
   ```
2. Place the **LeapGestRecog** dataset in the specified path.  
3. Run the training + demo script:
   ```bash
   python gesture_train.py
   ```
4. After training, the webcam window will open. Perform gestures in front of your camera, and the model will predict them live. Press **ESC** to exit.

---
