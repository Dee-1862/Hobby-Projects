# Customer Churn Prediction Project

## Overview

This project aims to predict customer churn for KKBOX using machine learning techniques. The project aims to predict if a user will **churn** (meaning the customer will not make a new service subscription transaction within 30 days after their current membership expires). This project involves data preprocessing, feature engineering, exploratory data analysis (EDA), model training, and deployment using a Streamlit web application.

The core of the project consists of two main Python scripts:

-   `main.py`: This script performs the data preprocessing, feature engineering, EDA, model training, and saving of the best-performing model.
-   `app.py`: This script creates an interactive web application using Streamlit, allowing users to input customer data and receive a churn prediction from the trained model.

## The Libraries Used

These are few of the Python libraries used in this project:
    -   `pandas`: For data manipulation and analysis.
    -   `plotly`: For creating interactive plots.
    -   `matplotlib`: For creating static plots.
    -   `seaborn`: For creating informative statistical graphics.
    -   `sklearn` (scikit-learn): For various machine learning tasks including model implementation (`LogisticRegression`, `GaussianNB`, `RandomForestClassifier`, `DecisionTreeClassifier`), and potentially data scaling and splitting (though custom functions are used here).
    -   `lightgbm`: For the LightGBM gradient boosting framework (`LGBMClassifier`).
    -   `joblib`: For saving and loading trained machine learning models.
    -   `streamlit`: For building the interactive web application.

## Project Structure

The final project directory structure is as follows:
```
├── data/
│   ├── raw/
│   │   ├── members_v3.csv
│   │   ├── transactions_v2.csv
│   │   ├── train_v2.csv
│   │   └── user_logs_v2.csv
│   └── processed/
│       └── data.csv
├── model/
│   └── model.sav
├── reusable.py
├── preprocessing.py
├── main.py
└── app.py
└── README.md
```

-   `data/raw/`: Contains the raw input datasets.
-   `data/processed/`: Stores the merged and preprocessed dataset (`data.csv`).
-   `model/`: Contains the saved trained machine learning model (`model.sav`).
-   `reusable.py`: Holds custom reusable functions for data processing and analysis.
-   `preprocessing.py`: Contains functions for preprocessing data specifically for model input.
-   `main.py`: The main script for data loading, preprocessing, feature engineering, EDA, model training, and saving.
-   `app.py`: The Streamlit application script for online churn prediction.
-   `README.md`: This file, providing an overview of the project.

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd CustomerChurnPrediction
    ```

2.  **Install required libraries:**
    It is recommended to create a virtual environment before installing the dependencies.
    ```bash
    python -m venv .venv
    .venv\Scripts\activate  # On Windows
    pip install -r requirements.txt
    ```

3.  **Ensure data files are in the `data/raw/` directory.** Download the `members_v3.csv`, `transactions_v2.csv`, `train_v2.csv`, and `user_logs_v2.csv` files from [**https://www.kaggle.com/competitions/kkbox-churn-prediction-challenge/data**] and place them in the `data/raw/` folder.

## Running the Project

### 1. Data Processing and Model Training

To preprocess the data, perform feature engineering, train the machine learning model, and save it, run the `main.py` script:

```bash
python main.py
```