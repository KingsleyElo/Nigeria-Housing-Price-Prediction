# Nigeria Housing Price Prediction

A machine learning project applying **regularized linear regression (Ridge)** to predict house prices from Nigerian real estate data scraped from Nigeria Property Centre.

---

## 📋 Project Overview

This project demonstrates core machine learning principles using a **manual implementation of Ridge regression** on a real-world Nigerian housing dataset.

Key highlights include:

- Manual implementation of **Ridge regression using the closed-form solution**
- **Three-way data split** (train / validation / test) for proper model selection
- **Hyperparameter tuning** using validation RMSE
- Retraining the final model on **combined training + validation data**
- Feature engineering for missing values, categorical data, and temporal features
- Log transformation of a highly skewed price target

**Final Model Performance**  
RMSE ≈ **0.87** on log-transformed house prices (strong linear baseline on noisy real-world data)

---

## 📊 Dataset

**Source**: Kaggle – Nigeria Real Estate Dataset  
**Records**: 1,668 properties  

### Key Features
- Bedrooms, Bathrooms, Toilets, Parking Spaces  
- Total Area, Covered Area (sqm)  
- Property Type (Detached Duplex, Block of Flats, etc.)  
- Location (State: Lagos, Abuja, Ogun, Oyo, etc.)  
- Listing duration (days on market)

---

## 🛠️ Methodology & Techniques

### 1️⃣ Data Preprocessing & Feature Engineering
- Parsed numeric values from text fields (e.g. `"850 sqm" → 850.0`)
- Handled missing values using:
  - Zero-imputation for numeric features
  - Explicit **missingness indicator variables**
- Engineered a temporal feature: `Listing_Duration_Days`
- Encoded categorical variables using one-hot encoding
- Applied **log(1 + price)** transformation to stabilize variance

---

### 2️⃣ Manual Ridge Regression Implementation

Ridge regression was implemented **from scratch** using the closed-form solution:

$w = (X^TX + \lambda I)^{-1}X^Ty$

Key implementation details:
- Explicit bias term handling
- L2 regularization to control multicollinearity
- No reliance on `scikit-learn` for model fitting

---

### 3️⃣ Model Training & Selection Strategy

The modeling pipeline followed standard supervised learning best practices:

1. Data split into:
   - **Training set (60%)**
   - **Validation set (20%)**
   - **Test set (20%)**
2. For each regularization strength (α):
   - Train the model on the training set
   - Evaluate RMSE on the validation set
3. Select the best α based on validation performance
4. **Retrain the final model once** on the combined training + validation data
5. Evaluate generalization performance **once** on the held-out test set

This approach preserves test set integrity and avoids data leakage.

---

## 📈 Results

| Model Configuration | RMSE (Train+Val) | RMSE (Test) |
|---------------------|-----------------|-------------|
| Ridge (full features) | 0.82 | **0.87** |

### Interpretation
The linear Ridge model provides a strong baseline but is limited by its inability to capture non-linear relationships common in housing markets. The results suggest that tree-based oror models that capture non-linear relationships may further improve performance.

---

## 🚀 Quick Start

1. **Clone the repository**
```bash
git clone https://github.com/KingsleyElo/Nigeria-Housing-Price-Prediction.git
cd nigeria-housing-prediction
```

2. **Install dependencies**
pip install -r requirements.txt

3. **Launch Jupyter Notebook**
jupyter notebook Nigeria_Housing_Price_Prediction.ipynb

## 🎯 Learning Outcomes

Through this project, I:

- Implemented Ridge regression mathematically from scratch
- Applied correct train / validation / test workflow
- Tuned hyperparameters using validation RMSE
- Managed real-world data challenges (missing values, messy text fields)
- Engineered meaningful domain features
- Interpreted log-scale error metrics in a practical context

## 📝 Author

**Kingsley Eloebhose**  
Machine Learning Student | Nigeria Housing Market Analysis  
[LinkedIn](https://www.linkedin.com/in/kingsley-eloebhose-77ab41379) | [Email](mailto:eloebhosekingsley@outlook.com)

---

*Built with ❤️ for learning Ridge regression on Nigerian real estate data*


