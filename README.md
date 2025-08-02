# Overview

PhytoPath AI is a deep learning-based plant disease detection system that classifies leaf images into different disease categories using a fine-tuned VGG19 model. The model is trained on the PlantVillage Dataset, achieving high accuracy in disease classification.
This project is built using TensorFlow, Streamlit, and Transfer Learning, making plant disease detection accurate, efficient, and accessible for farmers, researchers, and agricultural experts.

# Model & Algorithm

Model Used: VGG19 (Pretrained on ImageNet, Fine-Tuned) 

Dataset: PlantVillage Dataset link: https://www.kaggle.com/datasets/emmarex/plantdisease

Algorithm: Transfer Learning & Fine-Tuning

Loss Function: Categorical Crossentropy

Optimizer: Adam

Image Preprocessing:
  
  Resized images to 224x224
  
  Normalized pixel values to [0,1]
  
  Data Augmentation: Rotation, Zoom, Flip, Shear


