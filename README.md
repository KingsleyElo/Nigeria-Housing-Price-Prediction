# Nigeria Housing Price Prediction

A machine learning project applying **regularized linear regression (Ridge)** to predict house prices from Nigerian real estate data scraped from Nigeria Property Centre.

## 📋 Project Overview

This project demonstrates core machine learning concepts using **Ridge regression** on a real-world Nigerian housing dataset:

- **Custom Ridge implementation** with $X^TX + \lambda I$ regularization
- **Hyperparameter tuning** via validation RMSE
- **Feature engineering** for missing data and categorical variables
- **Log-transformation** of skewed price target
- **Train/validation/test splitting** with reproducible seeding

**Final Model Performance**: ~0.87 RMSE on log-transformed prices (reasonable baseline for linear models)

## 📊 Dataset

**Source**: [Kaggle - Nigeria Real Estate Dataset](https://www.kaggle.com/datasets/chik0di/nigeria-real-estate-dataset)  
**Records**: 1,668 properties  

**Key Features**
- Bedrooms, Bathrooms, Toilets, Parking Spaces
- Total Area, Covered Area (sqm)
- Property Type (Detached Duplex, Block of Flats, etc.)
- Location (State: Lagos, Abuja, Ogun, Oyo, etc.)
- Listing Duration (days on market)

## 🛠️ Key Techniques Applied

### 1. **Data Preprocessing**
- ✓ Missing value handling with indicators
- ✓ Area parsing (`"850 sqm"` → `850.0`)
- ✓ Date feature engineering (`ListingDurationDays`)
- ✓ One-hot encoding for Property Type and State
- ✓ Log transformation of target variable (`log(1 + Price)`)

### 2. **Model Implementation**
- Ridge Regression: $$w = (X^TX + \lambda I)^{-1}X^Ty$$
- Custom bias term and regularization matrix
- Grid search: $\alpha \in [10^{-5}, 10^{-3}, \ldots, 10]$

### 3. **Evaluation Pipeline**
- Train: 60% | Validation: 20% | Test: 20%
- Metric: RMSE on log-scale prices
- Numeric-only versus full-feature model comparison

## 📈 Results Summary

| Model                  | Train RMSE | Test RMSE |
|:----------------------:|:---------:|:---------:|
| Full Ridge (+categorical) | 0.82     | **0.88**  |


**Takeaway**: The linear model establishes a solid baseline (~50–100% price error), limited by dataset complexity and linear assumptions.


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

- Implemented Ridge regression from scratch (no `sklearn`)
- Handled real-world data issues (missing values, text parsing)
- Tuned regularization using a validation set
- Interpreted log-scale RMSE in a business context
- Compared numeric-only and full feature sets
- Visualized model diagnostics (residuals and predictions)

## 📝 Author

**Kingsley Eloebhose**  
Machine Learning Student | Nigeria Housing Market Analysis  
[LinkedIn](https://www.linkedin.com/in/kingsley-eloebhose-77ab41379) | [Email](mailto:eloebhosekingsley@outlook.com)

---

*Built with ❤️ for learning Ridge regression on Nigerian real estate data*


