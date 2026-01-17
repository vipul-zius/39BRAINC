"""
BRAINCELL - Brain Tumor Detection System
A deep learning based system for detecting brain tumors from MRI scans
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import cv2
import os
import warnings
warnings.filterwarnings('ignore')

class BrainTumorDetector:
    """Main class for brain tumor detection"""
    
    def __init__(self):
        self.model = None
        self.classes = ['No Tumor', 'Tumor Detected']
        self.input_shape = (224, 224, 3)
        
    def build_model(self):
        """Build CNN model for tumor detection"""
        model = keras.Sequential([
            keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Conv2D(64, (3, 3), activation='relu'),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Conv2D(128, (3, 3), activation='relu'),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Flatten(),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(2, activation='softmax')
        ])
        
        model.compile(optimizer='adam',
                     loss='sparse_categorical_crossentropy',
                     metrics=['accuracy'])
        
        self.model = model
        return model
    
    def preprocess_image(self, image_path):
        """Preprocess MRI image for model input"""
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        img = cv2.resize(img, (self.input_shape[0], self.input_shape[1]))
        img = img / 255.0  # Normalize
        return np.expand_dims(img, axis=0)
    
    def predict(self, image_path):
        """Make prediction on an MRI image"""
        processed_img = self.preprocess_image(image_path)
        predictions = self.model.predict(processed_img)
        class_idx = np.argmax(predictions[0])
        confidence = predictions[0][class_idx]
        
        return {
            'class': self.classes[class_idx],
            'confidence': float(confidence),
            'tumor_detected': class_idx == 1
        }
    
    def train_model(self, train_data, val_data, epochs=10):
        """Train the model on dataset"""
        history = self.model.fit(
            train_data,
            validation_data=val_data,
            epochs=epochs,
            verbose=1
        )
        return history

def main():
    """Main function to demonstrate the tumor detector"""
    print("=" * 50)
    print("BRAINCELL - Brain Tumor Detection System")
    print("=" * 50)
    
    # Initialize detector
    detector = BrainTumorDetector()
    
    # Build model
    print("\n[1] Building model architecture...")
    model = detector.build_model()
    model.summary()
    
    print("\n[2] Model ready for training!")
    print(f"   - Input shape: {detector.input_shape}")
    print(f"   - Classes: {detector.classes}")
    print(f"   - Total parameters: {model.count_params():,}")
    
    print("\n[3] Sample workflow:")
    print("   - Load MRI scan images")
    print("   - Preprocess and augment data")
    print("   - Train CNN model")
    print("   - Evaluate on test set")
    print("   - Deploy for clinical use")
    
    print("\n" + "=" * 50)
    print("System initialized successfully!")
    print("=" * 50)

if __name__ == "__main__":
    main()
