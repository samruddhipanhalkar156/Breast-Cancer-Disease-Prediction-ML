# 🎀 Breast Cancer Diagnostic Prediction System  

## 📌 Project Overview  
This project is a **machine learning-based diagnostic tool** that predicts whether a breast tumor is:  
- **Benign (0)** or  
- **Malignant (1)**  

based on the **UCI Breast Cancer Diagnostic dataset**.  

The pipeline includes:  
1. **Data Cleaning & Preprocessing**  
2. **Exploratory Data Analysis (EDA)**  
3. **Feature Selection (removing highly correlated features)**  
4. **Model Building & Hyperparameter Tuning** (Logistic Regression, SVM, Random Forest, XGBoost)  
5. **Model Evaluation & Comparison**  
6. **Interactive Web App with Streamlit**  

The app allows users to:  
✅ Select a trained model and view its performance metrics  
✅ Enter tumor feature values manually for prediction  
✅ Upload a CSV file for batch predictions  
✅ View binary outcomes (*Benign vs Malignant*)  

```

## 📂 Folder Structure  

Breast Cancer Prediction Analysis
│
├── Models
│ ├── LogisticRegression_binary.pkl
│ ├── RandomForest_binary.pkl
│ ├── SVM_binary.pkl
│ └── XGBoost_binary.pkl
│
├── Model Results
│ ├── model_results.xlsx
│ └── manifest.pkl
│
├── EDA Results
│ ├── correlation_heatmap.png
│ ├── boxplots.png
│ └── pairplot.png
│
├── Requirements
│ └── requirements.txt
│
├── main.py ← Training pipeline
├── app.py ← Streamlit web app
└── README.md

yaml
Copy code

```

## 🧹 Data Cleaning & Preprocessing  
- Dataset fetched from **UCI Repository (ID: 17)**.  
- Converted diagnosis labels:  
  - `B → 0` (Benign)  
  - `M → 1` (Malignant)  
- Removed highly correlated features (> 0.9 correlation):  
['perimeter1', 'area1', 'concave_points1',
'perimeter2', 'area2', 'radius3', 'texture3',
'perimeter3', 'area3', 'concave_points3']

yaml
Copy code
- Scaled features using **StandardScaler**.  
- Saved preprocessing objects (`scaler.pkl`) for reuse.  

---

## 📊 Column Descriptions (Generalized)  

Since column names are technical (e.g., `radius1`, `texture2`), we provide simplified placeholders:  

## 📊 Column Descriptions  

| Column Name           | General Description |
|-----------------------|----------------------|
| **radius1**           | Mean radius of cell nuclei |
| **texture1**          | Mean texture of cell nuclei |
| **smoothness1**       | Mean smoothness of cell nuclei |
| **compactness1**      | Mean compactness measurement |
| **concavity1**        | Mean concavity measurement |
| **symmetry1**         | Mean symmetry of cell nuclei |
| **fractal_dimension1**| Mean fractal dimension |
| **radius2**           | Standard error of radius |
| **texture2**          | Standard error of texture |
| **smoothness2**       | Standard error of smoothness |
| **compactness2**      | Standard error of compactness |
| **concavity2**        | Standard error of concavity |
| **concave_points2**   | Standard error of concave points |
| **symmetry2**         | Standard error of symmetry |
| **fractal_dimension2**| Standard error of fractal dimension |
| **texture3**          | Worst texture of cell nuclei |
| **smoothness3**       | Worst smoothness of cell nuclei |
| **compactness3**      | Worst compactness measurement |
| **concavity3**        | Worst concavity measurement |
| **symmetry3**         | Worst symmetry of cell nuclei |
| **fractal_dimension3**| Worst fractal dimension |
| **Diagnosis**         | Target variable (0 = Benign, 1 = Malignant) |


*(Dropped features listed above are excluded from the final model input.)*  



## 🤖 Models Implemented  
1. **Logistic Regression** – tuned regularization & solvers  
2. **Support Vector Machine (SVM)** – tested multiple kernels  
3. **Random Forest** – optimized estimators & tree depth  
4. **XGBoost** – tuned learning rate, max depth, subsampling  

✅ Best models saved as `.pkl` files for deployment.  



## 📈 Evaluation Metrics  
Each model evaluated using:  
- `Accuracy`  
- `Precision`  
- `Recall`  
- `F1-score`  
- `AUC-ROC `

📊 Results saved in:  
`Model Results/model_results.xlsx`  



## 💻 Streamlit Web App (app.py)  
Features:  
- Dropdown to choose a model (LR, SVM, RF, XGBoost)  
- Displays metrics for selected model  
- **Manual input** → Predict Benign / Malignant  
- **CSV upload** → Batch predictions + download option  
- Tooltips showing simplified column descriptions  



## 🚀 How to Run  

### 1️⃣ Install Dependencies  
```bash
cd "Breast Cancer Prediction Analysis"
pip install -r Requirements/requirements.txt
```
2️⃣ Run Training Script

python main.py
3️⃣ Launch Web App
```bash

streamlit run app.py
Then open 👉 http://localhost:8501
```

## 📦 Requirements
`pandas`, `numpy`

`scikit-learn`

`matplotlib`, `seaborn`

`xgboost`

`pickle`, `joblib`

`streamlit`

`openpyxl`

## 🔮 Future Enhancements
Feature importance visualization (SHAP/Permutation Importance)

Deploy on Streamlit Cloud / Heroku

API integration for medical systems

## 👩‍💻 Author

**Samruddhi Panhalkar**

📧 Email: samruddhipanhalkar156@gmail.com

🏫 Robotics and Artificial Intelligence

🌐 LinkedIn
