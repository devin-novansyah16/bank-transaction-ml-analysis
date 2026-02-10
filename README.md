# Bank Transaction ML Analysis

A comprehensive machine learning project for analyzing bank transaction data using clustering and classification techniques.

## 📋 Project Overview

This project implements a two-phase machine learning pipeline:
1. **Clustering** - Customer segmentation and anomaly detection
2. **Classification** - Transaction classification using Decision Tree

Dataset contains **2,512 bank transaction samples** with comprehensive attributes including transaction details, customer demographics, and behavioral patterns.

## 📊 Dataset Features

- **TransactionID**: Unique identifier for each transaction
- **AccountID**: Unique account identifier
- **TransactionAmount**: Transaction value
- **TransactionDate**: Date and time of transaction
- **TransactionType**: Credit or Debit
- **Location**: Geographic location (US cities)
- **DeviceID**: Device used for transaction
- **IP Address**: IPv4 address during transaction
- **MerchantID**: Unique merchant identifier
- **AccountBalance**: Account balance after transaction
- **Channel**: Transaction channel (Online, ATM, Branch)
- **CustomerAge**: Customer age
- **CustomerOccupation**: Customer profession
- **LoginAttempts**: Number of login attempts before transaction
- And more...

## 🎯 Project Goals

- Perform customer segmentation using clustering algorithms
- Detect anomalies in transaction patterns
- Build classification models for transaction categorization
- Provide insights into customer behavior and transaction security

## 📁 Project Structure

```
bank-transaction-ml-analysis/
├── [Clustering]_Submission_Akhir_BMLP_Devin_Novansyah.ipynb
├── [Klasifikasi]_Submission_Akhir_BMLP_Devin_Novansyah.ipynb
├── bank_transactions_data_edited.csv
├── data_clustering.csv
├── decision_tree_model.h5
├── model_clustering.h5
└── README.md
```

## 🛠️ Technologies Used

- **Python 3.x**
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computing
- **Scikit-learn** - Machine learning algorithms
- **TensorFlow/Keras** - Deep learning models
- **Jupyter Notebook** - Interactive analysis

## 📝 Notebooks

### 1. Clustering Notebook
Performs customer segmentation and anomaly detection using various clustering techniques.

**Key Steps:**
- Data loading and exploration
- Data preprocessing
- Feature engineering
- Clustering model training
- Results visualization and interpretation

### 2. Classification Notebook
Builds classification models using the clustering results.

**Key Steps:**
- Load clustered data
- Data splitting (training/testing)
- Decision Tree classifier training
- Model evaluation
- Performance metrics (accuracy, precision, recall F1-score)

## 🚀 Getting Started

### Prerequisites
- Python 3.7+
- Jupyter Notebook or JupyterLab
- Required libraries (see Technologies Used)

### Installation

1. Clone the repository
```bash
git clone https://github.com/devin-novansyah16/bank-transaction-ml-analysis.git
cd bank-transaction-ml-analysis
```

2. Install required dependencies
```bash
pip install pandas numpy scikit-learn tensorflow jupyter
```

3. Run Jupyter Notebook
```bash
jupyter notebook
```

4. Open the notebooks in order:
   - Start with `[Clustering]_Submission_Akhir_BMLP_Devin_Novansyah.ipynb`
   - Then proceed to `[Klasifikasi]_Submission_Akhir_BMLP_Devin_Novansyah.ipynb`

## 📈 Results

The project includes trained models and results:
- **model_clustering.h5** - Trained clustering model
- **decision_tree_model.h5** - Trained classification model
- **data_clustering.csv** - Clustered dataset for classification

## 📧 Contact

**Author:** Devin Novansyah  
**Email:** devinnovansyah1611@gmail.com

## 📄 License

This project is open source and available under the MIT License.

---

*Last Updated: February 11, 2026*
