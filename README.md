# Tomato Leaf Disease Classification

This project implements a deep learning-based solution for classifying tomato leaf diseases using the **MobileNetV2** model. The solution focuses on detecting and identifying six common tomato leaf diseases, enabling early diagnosis and actionable insights for farmers.

## Table of Contents
1. [Overview](#overview)
2. [Features](#features)
3. [Dataset](#dataset)
4. [Technologies Used](#technologies-used)
5. [Model Architecture](#model-architecture)
6. [Web Application](#web-application)
7. [How to Run the App](#how-to-run-the-app)
8. [Results](#results)
9. [Future Work](#future-work)
10. [Acknowledgments](#acknowledgments)

## Overview
The **Tomato Leaf Disease Classification** application classifies images into one of six disease categories:
- Bacterial Spot
- Early Blight
- Healthy
- Late Blight
- Leaf Mold
- Yellow Leaf Curl Virus

It provides accurate predictions using transfer learning, featuring a lightweight MobileNetV2 architecture.

## Features
- **Disease Detection**: Classifies tomato leaf images into one of six categories.
- **Pre-Trained Model**: Utilizes the MobileNetV2 model for efficient and accurate predictions.
- **Visualization**: Includes heatmaps, bar charts, and classification reports for model evaluation.

## Dataset
- The dataset contains annotated images for six tomato leaf conditions.
- Images are resized to 128x128 pixels for uniformity.
- Data augmentation techniques were applied to improve variability and robustness.

## Technologies Used
- **Python**: Programming language for data preprocessing and model development.
- **TensorFlow & Keras**: Deep learning libraries for building the classification model.
- **OpenCV**: Image preprocessing.
- **Matplotlib & Seaborn**: Visualization libraries for generating plots and heatmaps.
- **Streamlit**: For building the web application.
- **Sklearn**: For model evaluation metrics like confusion matrices and classification reports.

## Model Architecture
### MobileNetV2
- A pre-trained MobileNetV2 model is used as the base architecture, excluding the top layers.
- Layers:
  - GlobalAveragePooling2D: Reduces the dimensions of the feature maps.
  - Dense: Output layer for class predictions (6 classes).

### Training Details
- Optimizer: Adam
- Loss Function: Sparse Categorical Crossentropy
- Metrics: Accuracy
- Input Image Shape: 128x128x3 (RGB images)

## Web Application
### Features
- **User-Friendly Interface**: Built with Streamlit.
- **Disease Selection**: Dropdown for selecting suspected diseases.
- **Image Upload**: Users can upload tomato leaf images in JPG, JPEG, or PNG formats.
- **Prediction Results**: Displays predicted disease, confidence scores, and comparison with user input.
- **Visualization**: Confidence scores are shown using bar charts.

### How It Works
1. Users select a suspected disease from the dropdown menu.
2. Upload an image of a tomato leaf.
3. The app preprocesses the image and uses the trained MobileNetV2 model for predictions.
4. Results and confidence scores are displayed, along with a visual comparison.

### Sample Web App Screenshots
![Home Page](https://github.com/Mekapothulavenu/tomato_leaf-_disease_classification/blob/51e2fa7e9d2cb9574018a92807cfeda5151f2b3f/app_and_code_images/streamlit_app3.png)
![upload](https://github.com/Mekapothulavenu/tomato_leaf-_disease_classification/blob/51e2fa7e9d2cb9574018a92807cfeda5151f2b3f/app_and_code_images/streamlit_app4.png)
![Prediction Results](https://github.com/Mekapothulavenu/tomato_leaf-_disease_classification/blob/51e2fa7e9d2cb9574018a92807cfeda5151f2b3f/app_and_code_images/streamlit_app5.png)
![Confidence Scores](https://github.com/Mekapothulavenu/tomato_leaf-_disease_classification/blob/51e2fa7e9d2cb9574018a92807cfeda5151f2b3f/app_and_code_images/streamlit_app6.png)

## How to Run the App
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/tomato-leaf-disease-classification.git
   cd tomato-leaf-disease-classification
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Place the dataset in the specified directory.
4. Train the model:
   ```bash
   python train.py
   ```
5. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```
6. Open the application in your browser at `http://localhost:8501`.

## Results
| Metric            | Value   |
|-------------------|---------|
| Accuracy          | 93%     |
| Precision         | 0.92    |
| Recall            | 0.92   |
| F1-Score          | 0.92    |

### Confusion Matrix
- Visualized as a heatmap to show classification performance across all classes.

## Future Work
- Expand the dataset to include more plant species and diseases.
- Integrate real-time disease detection using live video feeds.
- Experiment with advanced architectures like Vision Transformers.
- Add additional features like multilingual support in the web app.

## Acknowledgments
- **TensorFlow and Keras Documentation**
- **PlantVillage Dataset**
- **Python Community**

This project demonstrates the potential of deep learning in revolutionizing agriculture by providing accessible and efficient solutions for disease detection.
