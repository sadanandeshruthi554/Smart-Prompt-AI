## Smart-Prompt-AI
## 📌 Project Overview

This project demonstrates the practical application of Large Language Models (LLMs) using advanced prompting techniques:

- Chain-of-Thought (CoT) Prompting for Customer Review Sentiment Analysis
- Tree-of-Thought (ToT) Prompting for Machine Learning Hyperparameter Evaluation

The solution consists of two interactive Streamlit applications that leverage modern LLMs to perform intelligent reasoning and decision-making tasks. The project was developed as part of a Generative AI assignment focused on prompt engineering and explainable AI workflows.
-----------------
## 🚀 Features
### 1️⃣ Sentiment Analysis using Chain-of-Thought (CoT)
- Analyze customer reviews using Gemini LLM.
- Detect positive, negative, and neutral sentiments.
- Explain sentiment predictions with step-by-step reasoning.
- Adjustable generation parameters:
    - Temperature
    - Maximum Output Tokens
- Interactive Streamlit interface.

### 2️⃣ Hyperparameter Optimization using Tree-of-Thought (ToT)
- Train and tune a Random Forest Regression model on AQI data.
- Generate multiple hyperparameter configurations using RandomizedSearchCV.
- Use Groq LLM to analyze candidate configurations.
- Compare:
    - Mean Test Score
    - Standard Deviation
    - Test R² Score
    - Training Time
    - Rank Test Score
- Recommend the best configuration with justification.
------------------
## 🛠️ Tech Stack
- Python
- Streamlit
- Google Gemini API
- Groq API
- Scikit-Learn
- Pandas
- NumPy
- Matplotlib
- Seaborn
- RandomizedSearchCV
---------------
## 📂 Project Structure
├── app_task1.py                 # Sentiment Analysis (CoT)
├── app_task2.py                 # Hyperparameter Analysis (ToT)
├── ml.py                        # AQI Model Training & Tuning
├── AQI.csv                      # Dataset
├── AQI_Model_Results.csv        # Hyperparameter Results
├── requirements.txt             # Dependencies
├── .env                         # API Keys (Not Uploaded)
└── README.md


