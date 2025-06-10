# **End-to-End Lead Scoring ML Pipeline**

This project demonstrates a complete machine learning pipeline for a lead scoring model. The goal is to predict the probability of a lead converting into a customer, allowing a sales team to prioritize their efforts on the most promising leads. The entire environment is containerized using Docker for easy setup and reproducibility.

## **🚀 Key Features**

* **Data Ingestion:** Loads lead data from a PostgreSQL database.  
* **End-to-End Pipeline:** Covers all stages from data cleaning and EDA to feature engineering and modeling.  
* **Predictive Model:** Utilizes a Logistic Regression model to calculate a lead conversion score.  
* **High Performance:** Achieves **\~94% accuracy** and an **AUC score of \~0.98** on the test set.  
* **Reproducibility:** Fully containerized with Docker and Docker Compose for a one-command setup.  
* **Secure Configuration:** Uses environment variables for managing sensitive credentials, ready for public repositories.

## **🛠️ Tech Stack**

* **Backend & Data:** PostgreSQL, Docker  
* **ML & Data Science:** Python, Pandas, Scikit-learn, Jupyter Lab  
* **Orchestration:** Docker Compose  
* **Environment Management:** python-dotenv

## **📊 Pipeline Overview**

1. **Database Setup:** A PostgreSQL database is orchestrated via Docker Compose to store the raw lead data.  
2. **Data Ingestion:** A Python script connects to the database to fetch the data into a Pandas DataFrame.  
3. **Data Cleaning & EDA:** Missing values are handled with a combination of dropping low-utility columns and strategic imputation (e.g., mode, median, 'Not Specified' category).  
4. **Feature Engineering:** Categorical features are one-hot encoded and numerical features are scaled. Low-variance features are automatically removed.  
5. **Model Training:** A Logistic Regression model is trained on the processed data.  
6. **Model Evaluation:** The model's performance is evaluated using accuracy, precision, recall, F1-score, and ROC AUC.  
7. **Lead Scoring:** The trained model is used to generate a conversion probability score (from 0 to 100\) for each lead.

## **⚙️ Setup and Installation**

**Prerequisites:** Docker and Docker Compose must be installed on your system.

**1\. Clone the Repository:**

```bash
git clone https://github.com/1bytess/lead-scoring-pipeline.git
cd lead-scoring-pipeline
```

2\. Create the Environment File:  
This project uses an .env file to manage credentials securely. Create a file named .env in the root directory and add the following, using the same credentials as your docker-compose.yml:  
```
DB_USER=your_db_user  
DB_PASSWORD=your_db_password
DB_HOST=your_db_host  
DB_PORT=5432  
DB_NAME=your_db_name
```

*(Note: A .gitignore file is included to prevent this file from being uploaded to GitHub.)*

3\. Launch the Services:  
From the root directory, run the following command to build and start all the services (PostgreSQL, pgAdmin, Jupyter Lab):  
```
docker-compose up --build -d
```
**4\. Run the Pipeline:**

* Access Jupyter Lab by navigating to http://localhost:8888 in your browser.  
* Open the .ipynb notebook file.  
* Run the cells from top to bottom to execute the entire data pipeline and model training process.

## **📈 Results**

The final Logistic Regression model demonstrated excellent performance on the held-out test data:

* **Accuracy:** 94.15%  
* **ROC AUC Score:** 0.9813  
* **Precision (Converted):** 0.95  
* **Recall (Converted):** 0.89

These results indicate a highly reliable model that can effectively distinguish between converting and non-converting leads.

## **🔮 Future Work & Deployment Roadmap**

* **Advanced Feature Engineering:**  
  * \[ \] Implement advanced feature selection techniques like **Recursive Feature Elimination (RFE)** to identify the most impactful features and potentially simplify the model.  
* **Model Optimization:**  
  * \[ \] Perform **hyperparameter tuning** using GridSearchCV or RandomizedSearchCV to further optimize model performance.  
  * \[ \] Experiment with more complex models like **Random Forest** or **Gradient Boosting (XGBoost)** to compare results.  
* **Deployment:**  
  * \[ \] **Real-time Scoring API:** Package the trained model and preprocessing pipeline into a REST API using **Flask** or **FastAPI**. This would allow other applications to get lead scores on-demand.  
  * \[ \] **Interactive Dashboard:** Develop a dashboard using **Streamlit** or **Dash** to visualize lead scores, explore model predictions, and monitor performance in a user-friendly interface.
