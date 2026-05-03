# 🌸 Iris Flower Classification (Deep Learning Project)

This repository contains a production-grade implementation of an Iris species classifier. It utilizes a **Deep Learning** approach for high-accuracy classification while maintaining interpretability through feature importance analysis.

---

## 📌 Overview

The project follows a **modular machine learning pipeline architecture**, ensuring clean code and scalability. 

### Key Components:
- **Data Pipeline:** Automated ingestion and transformation.
- **Hybrid Modeling:** Neural Networks for prediction + Random Forest for interpretability.
- **Dual Deployment:** Integrated web interfaces using both **Flask** and **Streamlit**.

---

## 🚀 Project Objectives

* **Preprocess & Scale:** Handle the Iris dataset features using standard scaling techniques.
* **Deep Learning:** Architect a robust Neural Network for multi-class classification.
* **Interpretability:** Generate feature importance plots to understand model decisions.
* **Evaluation:** Provide performance metrics including confusion matrices.
* **Web UI:** Deploy the model for real-time predictions.

---

## 🧠 Problem Statement

The goal is to accurately predict one of three species of the Iris flower based on its physical measurements:
- **Setosa**
- **Versicolor**
- **Virginica**

---

## ⚙️ Tech Stack

* **Language:** Python (3.10 – 3.12)
* **Modeling:** TensorFlow / Keras, Scikit-learn
* **Data Processing:** Pandas, NumPy
* **Visualization:** Matplotlib, Seaborn
* **Deployment:** Flask, Streamlit
* **Environment:** VS Code, Git/GitHub

---

## 🗂️ Project Structure

```text
Iris-Flower-Classification/
<<<<<<< HEAD
│
├── artifacts/                        # ML outputs
│   ├── data.csv
│   ├── train.csv
│   ├── test.csv
│   ├── model.h5
│   └── preprocessor.pkl
│
├── notebook/                         # Jupyter notebooks
│   ├── data/
│   │   └── iris.csv
│   ├── EDA.ipynb
│   └── MODEL_TRAINING.ipynb
│
├── src/                              # Core ML pipeline (package)
│   ├── components/
│   │   ├── __init__.py
│   │   ├── data_ingestion.py
│   │   ├── data_transformation.py
│   │   └── model_trainer.py
│   │
│   ├── pipeline/
│   │   ├── __init__.py
│   │   ├── predict_pipeline.py
│   │   └── training_pipeline.py
│   │
│   ├── __init__.py
=======
├── artifacts/              # Generated files (models & outputs)
│   ├── model.h5
│   ├── preprocessor.pkl
│   ├── confusion_matrix.png
│   └── feature_importance.png
│
├── notebook/               # Jupyter notebooks (EDA & training)
│   ├── data/
│   │   └── iris.csv
│   ├── 1. EDA IRIS DATASET.ipynb
│   └── 2. MODEL TRAINING IRIS.ipynb
│
├── src/                    # Core ML pipeline code
│   ├── components/
│   │   ├── data_ingestion.py
│   │   ├── data_transformation.py
│   │   └── model_trainer.py
│   ├── pipeline/
│   │   ├── training_pipeline.py
│   │   └── predict_pipeline.py
>>>>>>> d743ec0794036bbb46ae63f1cece06729a8a3539
│   ├── exception.py
│   ├── logger.py
│   └── utils.py
│
<<<<<<< HEAD
├── static/                           # Frontend assets
│   ├── css/
│   │   ├── style.css
│   │   └── style_1.css
│   │
│   └── images/
│       ├── setosa.jpg
│       ├── versicolor.jpg
│       └── virginica.jpg
│
├── templates/                        # Flask templates
│   ├── home.html
│   └── index.html
│
├── logs/                             # Logs (empty in GitHub)
│   └── .gitkeep
│
├── app.py                            # Flask entry point
├── requirements.txt                  # Dependencies
├── setup.py                          # Package config
├── README.md                         # Documentation
├── .gitignore                        # Ignore rules
=======
├── templates/              # Flask UI
│   └── index.html
│
├── static/                 # Graphs & images
│   ├── confusion_matrix.png
│   └── feature_importance.png
│
├── app.py                  # Flask application
├── streamlit_app.py        # Streamlit application
├── requirements.txt
├── README.md
└── .gitignore
>>>>>>> d743ec0794036bbb46ae63f1cece06729a8a3539

## ⚙️ Tech Stack

* **Python**
* **TensorFlow / Keras (Deep Learning)**
* **Scikit-learn (Preprocessing + Feature Importance)**
* **Pandas, NumPy**
* **Matplotlib, Seaborn**
* **Flask & Streamlit (Deployment)**

---

## 🔄 ML Pipeline Workflow

### 1. Data Ingestion
- Reads Iris dataset  
- Splits into train and test data  
- Stores dataset  

---

### 2. Data Transformation
- Label encoding of target variable  
- Feature scaling using StandardScaler  
- Saves preprocessing pipeline (`preprocessor.pkl`)  

---

### 3. Model Training (Deep Learning)
- Neural Network using Keras  
- Dense layers with ReLU activation  
- Softmax output layer  
- Saves trained model (`model.h5`)  

---

### 4. Model Evaluation
- Accuracy score  
- Confusion matrix  
- Feature importance using Random Forest  

---

## 📊 Features Used

### Input Features
- Sepal Length  
- Sepal Width  
- Petal Length  
- Petal Width  

### Target Variable
- Species (Setosa, Versicolor, Virginica)

---

## 🧪 How to Run the Project

Step 1: Clone the Repository :

```bash
git clone https://github.com/VighaneshThakare/iris-flower-classification.git
cd iris-flower-classification


Step 2: Create Virtual Environment :

```bash
py -3.12 -m venv venv
venv\Scripts\activate


Step 3: Install Dependencies :

``` bash
pip install -r requirements.txt


Step 4: Train the Model (Important) :

``` bash
python -m src.pipeline.training_pipeline


----> This will generate the following files inside the artifacts/ folder:

artifacts/
 ├── model.h5
 ├── preprocessor.pkl
 ├── confusion_matrix.png
 └── feature_importance.png


Step 5: Run the Application
▶️ Streamlit App

``` bash
streamlit run streamlit_app.py


▶️ Flask App

``` bash
python app.py

Open in browser:
http://127.0.0.1:5000

Team-
Vighanesh Thakare
Siddharth Sharma
Somesh Pal
Aditya Nikam
