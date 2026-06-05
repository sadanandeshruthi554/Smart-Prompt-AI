## Smart-Prompt-AI
## 📌 Project Overview

This project demonstrates the practical application of Large Language Models (LLMs) using advanced prompting techniques:

- Chain-of-Thought (CoT) Prompting for Customer Review Sentiment Analysis
- Tree-of-Thought (ToT) Prompting for Machine Learning Hyperparameter Evaluation

    The solution consists of two interactive Streamlit applications that leverage modern LLMs to perform intelligent reasoning and decision-making tasks. The project was developed as part of a Generative AI          assignment focused on prompt engineering and explainable AI workflows.
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
- app_task1.py           --->    # Sentiment Analysis (CoT)
- app_task2.py           --->    # Hyperparameter Analysis (ToT)
- ml.py                  --->    # AQI Model Training & Tuning
- AQI.csv                --->    # Dataset
- AQI_Model_Results.csv  --->    # Hyperparameter Results
- requirements.txt       --->    # Dependencies
- .env                   --->    # API Keys (Not Uploaded)
- README.md
-----------------
## 🔄 Workflow
### Task 1: Sentiment Classification
1. User enters a customer review.
2. Gemini LLM analyzes:
    - Positive phrases
    - Negative phrases
    - Mixed opinions
    - Overall sentiment
3. Final sentiment is classified as:
    - Positive
    - Negative
    - Neutral
4. Reasoning is generated using CoT prompting.

### Task 2: Hyperparameter Selection
1. AQI dataset is preprocessed.
2. Random Forest model is trained.
3. RandomizedSearchCV evaluates multiple configurations.
4. Results are exported to CSV.
5. Groq LLM applies Tree-of-Thought reasoning.
6. Best hyperparameter configuration is selected and justified.
-----------------
## ⚙️ Installation
### Clone Repository

    git clone https://github.com/your-username/llm-sentiment-hyperparameter-assistant.git
    cd llm-sentiment-hyperparameter-assistant

### Install Dependencies

    pip install -r requirements.txt

### Configure Environment Variables

- Create a .env file:

    - GEMINI_API_KEY=your_gemini_api_key
    - GROQ_API_KEY=your_groq_api_key
---------------------
## ▶️ Run Applications
### Sentiment Analysis App

    streamlit run app_task1.py

### Hyperparameter Analysis App

    streamlit run app_task2.py
-----------------------
## 📊 Dataset

The project uses an Air Quality Index (AQI) dataset for machine learning model training and hyperparameter optimization. The dataset includes environmental indicators such as:

- PM2.5
- PM10
- NO₂
- SO₂
- CO
- NOx
- City

    These features are used to predict AQI values using a Random Forest Regressor.
-----------------------
## 🎯 Learning Outcomes
- Prompt Engineering with CoT and ToT techniques.
- LLM-driven reasoning and decision support.
- Machine Learning model optimization.
- Explainable AI workflows.
- Streamlit application development.
- Integration of Gemini and Groq APIs.
--------------------
## ✅ Conclusion

- This project demonstrates the effective use of Generative AI and Prompt Engineering techniques by implementing Chain-of-Thought (CoT) and Tree-of-Thought (ToT) reasoning frameworks. The sentiment analysis application provides explainable sentiment predictions from customer reviews, while the hyperparameter optimization assistant intelligently evaluates machine learning configurations and recommends the best-performing model. By integrating Google Gemini, Groq LLM, Scikit-Learn, and Streamlit, the project showcases how Large Language Models can enhance decision-making, improve interpretability, and support real-world AI applications through structured reasoning and explainable outputs.





