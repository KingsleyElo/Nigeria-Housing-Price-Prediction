# Nigeria Housing Price Prediction 🚀

This project implements a linear regression model using a **Ridge regression pipeline built with scikit-learn** to predict Nigerian house prices, achieving an approximately **24% RMSE improvement (0.88 → 0.67)** over a carefully implemented manual Ridge regression model through **feature scaling, numerically stable solvers, and cross-validated hyperparameter tuning**.

The repository compares a **from-scratch Ridge regression implementation (linear algebra)** with an **sklearn-based Pipeline approach**, illustrating how factors such as scaling, solver choice, and validation strategy can significantly influence model performance—even when both implementations are methodologically sound and free from data leakage.

---

## 🎯 Project Highlights

| Aspect | Manual Implementation | Sklearn Pipeline |
|------|----------------------|------------------------------------|
| Test RMSE | 0.88 | **0.67** |
| Scaling | ❌ None | ✅ **StandardScaler** |
| Validation | Single split | **5-fold cross-validation** |
| Regularization α | 0.0001 | **10 (CV-tuned)** |

---

## 📋 What This Project Demonstrates

- Manual Ridge regression using the closed-form solution **(XᵀX + λI)⁻¹Xᵀy**
- Why **feature scaling** drastically changes the optimal regularization strength
- How **missing-value indicators** allow linear models to learn from absence of data
- Proper **train / validation / test** workflows using cross-validation without data leakage

---

## 📊 Dataset

- **Source:** Kaggle – [Nigeria Property Centre dataset](https://www.kaggle.com/datasets/chik0di/nigeria-real-estate-dataset).
- **Size:** **1,668 residential property listings** across Lagos, Abuja, Ogun, Oyo, and other states

| Feature Type | Examples |
|-------------|----------|
| Numeric | Bedrooms, Bathrooms, Area (sqm), Listing Duration |
| Categorical | Property Type, District, State, Servicing, Furnishing |
| Target | **log1p(Price)** to stabilize extreme right skew |

---

## 🛠 Feature Engineering

- Parsed area strings such as `"850 sqm"` into numeric floats  
- Created **listing duration (days)** from date differences  
- Added **missingness indicator columns** for key numeric features  
- One-hot encoded categorical features (capturing location and property-type premiums)  
- Embedded all preprocessing inside an sklearn **Pipeline** to ensure consistent treatment of train and test data  

---

## ⚙️ Model Implementations

Two Ridge regression implementations are included:

- **Manual Ridge Regression**  
  Implemented using linear algebra to illustrate the effect of L2 regularization on the normal equation.

- **Sklearn Ridge Pipeline**  
  Uses preprocessing, scaling, and Ridge regression with **GridSearchCV** and **5-fold cross-validation**.

---

## 🔁 Training & Evaluation Workflow

1. Split data into **80% training** and **20% held-out test**
2. Tune regularization strength **α** using cross-validation on training data only
3. Select the best model and evaluate **once** on unseen test data
4. Final test performance achieves **0.67 RMSE** on log-transformed prices

This workflow avoids data leakage and mirrors real-world ML deployment practices.

---

## 📈 Results & Interpretation

| Model | Validation / CV RMSE | Test RMSE |
|------|---------------------|----------|
| **Sklearn Ridge Pipeline** | **0.79** | **0.67** |
| Manual Ridge Regression | 0.87 | 0.88 |

---

## 🧠 Why the Sklearn Pipeline Performs Better

Although both approaches use Ridge regression, the sklearn pipeline generalizes better due to:

- Proper **feature scaling before regularization**
- **Cross-validated α selection** instead of a single validation split
- Numerically stable solvers
- Identical preprocessing applied to unseen data during prediction

This highlights why production pipelines outperform mathematically correct but isolated implementations.

---

## 🧰 Tech Stack

- Python 3.9+
- pandas, numpy
- scikit-learn
- matplotlib, seaborn
- Ridge Regression, GridSearchCV

---

## 🎓 Skills Demonstrated

- Linear algebra & regularization theory  
- scikit learn pipeline implementation 
- Feature engineering with domain knowledge  
- Cross-validation and model selection  
- Debugging train–test performance gaps 
- Translating ML metrics into business insight  

---

## 👨‍💻 Author

**Kingsley Eloebhose**  
*Machine Learning Engineer* | *Real Estate Analytics*

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?logo=linkedin)](www.linkedin.com/in/kingsley-eloebhose-77ab41379)  
[![Email](https://img.shields.io/badge/Email-Contact-gray?logo=gmail)](mailto:eloebhosekingsley@outlook.com)

