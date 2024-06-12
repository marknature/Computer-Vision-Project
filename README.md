# Computer-Vision-Project
## Artifictual Intelligence Intern - Medical Image Classification

<p align="center">
  <img src="interncareers_logo.jpeg" alt="logo">
</p>

### Project Description
This project focuses on developing a deep learning model for image classification to diagnose medical conditions using chest X-ray images. The goal is to classify images as either normal or pneumonia.

### Project Structure
- **data/**: Contains training, testing, and validation datasets.
- **models/**: Stores the trained model.
- **notebooks/**: Jupyter notebooks for data preprocessing and visualization.
- **results/**: Contains evaluation metrics.
- **src/**: Source code for data preprocessing, model training, and evaluation.
- **README.md**: Project description and setup instructions.
- **requirements.txt**: List of required dependencies.

### Project Directory Structure
```
medical_image_classification/
│
├── data/
│   ├── train/
│   ├── test/
│   └── val/
│
├── models/
│   └── model.h5
│
├── notebooks/
│   └── data_preprocessing.ipynb
│
├── results/
│   └── evaluation_metrics.txt
│
├── src/
│   ├── data_preprocessing.py
│   ├── model_training.py
│   ├── model_evaluation.py
│   └── utils.py
│
├── README.md
└── requirements.txt
```

### Setup Instructions
1. **Install Dependencies**:
   ```sh
   pip install -r requirements.txt
   ```

2. **Download and Prepare Dataset**:
   Download the Chest X-ray Images (Pneumonia) dataset from [Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) and place it in the `data/` directory.

3. **Preprocess Data**:
   Run the Jupyter notebook `data_preprocessing.ipynb` to preprocess the images.

4. **Train the Model**:
   ```sh
   python src/model_training.py
   ```

5. **Evaluate the Model**:
   ```sh
   python src/model_evaluation.py
   ```

6. **View Results**:
   Evaluation metrics will be saved in `results/evaluation_metrics.txt`.
```

### `requirements.txt`
```plaintext
tensorflow
numpy
matplotlib
pillow
streamlit
scikit-learn
```

#### Instructions to Run the System

1. **Install Dependencies**:
   ```sh
   pip install -r requirements.txt
   ```

2. **Download and Prepare Dataset**:
   Download the Chest X-ray Images (Pneumonia) dataset from [Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) and place it in the `data/` directory.

3. **Preprocess Data**:
   Run the Jupyter notebook `data_preprocessing.ipynb` to preprocess the images.

4. **Train the Model**:
   ```sh
   python src/model_training.py
   ```

5. **Evaluate the Model**:
   ```sh
   python src/model_evaluation.py
   ```

6. **View Results**:
   Evaluation metrics will be saved in `results/evaluation_metrics.txt`.
   
