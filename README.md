# Multimodal Authentication System

This project is a machine learning-based multimodal authentication system that uses facial landmarks to classify and evaluate individuals across multiple labels. The system includes feature extraction, visualization, classification, and hyperparameter tuning using k-Nearest Neighbors (kNN) and Support Vector Machines (SVM). It also provides a web interface and a demo script for testing.

## Project Structure

- **src/main.py**: Main script to run the pipeline for training and evaluation.
- **src/web_app.py**: A Streamlit-based web application to test and visualize the system in real-time.
- **src/demo.py**: Script for demonstrating the functionality of the trained models with sample inputs.
- **src/evaluation/visualize_features.py**: Contains functions to visualize the feature space and confusion matrices.
- **src/models/**: Directory containing pre-trained models or saved models after training.
- **outputs/**: Directory where visualizations, confusion matrices, and saved models are stored.

## Requirements

Ensure you have the following dependencies installed before running the project:

- Python 3.9+
- Required libraries (Install using `pip install -r requirements.txt`)

## How to Run

### 1. **Run the Pipeline**
To train and evaluate the models:

```bash
python src/main.py
```

This will:
- Load images and extract landmarks.
- Visualize the feature space and save it as `outputs/feature_space.png`.
- Train and evaluate kNN and SVM classifiers.
- Save confusion matrices as `outputs/knn_confusion_matrix.png` and `outputs/svm_confusion_matrix.png`.
- Save the trained SVM model as `outputs/models/svm_model.pkl`.

### 2. **Run the Web Application**
To start the web application for real-time testing and visualization:

```bash
streamlit run src/web_app.py
```

The web application allows you to:
- Upload new images.
- Visualize extracted features.
- Test the classifiers using the trained model.

### 3. **Run the Demo**
To test the trained model with sample data:

```bash
python src/demo.py
```

This will:
- Load the pre-trained model.
- Run predictions on sample inputs.
- Display the results.

## Outputs

- **Feature Space Visualization**: `outputs/feature_space.png`
- **kNN Confusion Matrix**: `outputs/knn_confusion_matrix.png`
- **SVM Confusion Matrix**: `outputs/svm_confusion_matrix.png`
- **Saved SVM Model**: `outputs/models/svm_model.pkl`

## Notes

1. Ensure all directories (`outputs/` and `outputs/models/`) exist before running the scripts, or create them manually to avoid errors.
2. The system uses PCA for dimensionality reduction and hyperparameter tuning for optimized results.
3. The web application requires **Streamlit**. Install it using:

   ```bash
   pip install streamlit
   ```

---
