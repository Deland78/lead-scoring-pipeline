# **End-to-End Lead Scoring ML Pipeline & API**

This project demonstrates a complete, production-ready machine learning pipeline for a lead scoring model. The primary goal is to predict the probability of a lead converting into a customer, allowing a sales team to prioritize their efforts effectively.

The entire environment is containerized using Docker, and the final model is deployed as a REST API for real-time predictions.

## **🚀 Key Features**

* **Real-Time API:** Deployed with Flask and Gunicorn for live, on-demand lead scoring.  
* **End-to-End Pipeline:** Covers all stages from data cleaning and EDA to feature engineering and deployment.  
* **High Performance:** The underlying Logistic Regression model achieves **~94% accuracy** and an **AUC score of ~0.98**.  
* **Reproducibility:** Fully containerized with Docker and Docker Compose for a one-command setup of the entire application stack.  
* **Secure Configuration:** Uses environment variables for managing sensitive database credentials.  
* **Clean Architecture:** The project is organized into distinct app, models, and notebooks directories for clarity and maintainability.

## **🛠️ Tech Stack**

* **API & Backend:** Python, Flask, Gunicorn, PostgreSQL  
* **ML & Data Science:** Pandas, Scikit-learn, Jupyter Lab  
* **Orchestration & Deployment:** Docker, Docker Compose

## **📂 Project Structure**
```
lead-scoring-pipeline/  
├── app/                      # Contains the Flask API source code and Docker files
│   ├── app.py  
│   ├── docker-compose.yml    # Orchestrates all services  
│   ├── Dockerfile            # Defines the API service container  
│   ├── requirements.txt      # Python dependencies
│   └── gunicorn_config.py  
├── models/                   # Stores trained model artifacts  
│   ├── log_reg_model.joblib  
│   └── preprocessor.joblib  
├── notebooks/                # Holds the development notebook for experimentation  
│   ├── jupyter_db_connection_tes.py
│   └── lead_scoring_model_pipeline.ipynb  
├── .gitignore  
└── README.md  
```

## **⚙️ Local Setup and Usage**

**Prerequisites:** Docker and Docker Compose must be installed on your system.

**1. Clone the Repository:**
```
git clone https://github.com/1bytess/lead-scoring-pipeline.git  
cd lead-scoring-pipeline
```

2. Create the Environment File:  
The database service requires an .env file for credentials. Create a file named .env in the root directory and add the following:  
```
DB_USER=<user>
DB_PASSWORD=<your_database_password>
DB_HOST=<your_database_host> 
DB_PORT=5432  
DB_NAME=<your_database_name>
```

3. Launch the Application Locally:  
From the root directory, run the following command to build and start the API service, the database, and all other components:  
```
docker-compose up --build -d
```
The API will be running and available at `http://localhost:<your-port>`. *(Check your docker-compose.yml for the port you mapped for the app service).*

## **💡 How to Use the API**

You can send a POST request with lead data to the /predict endpoint to get a real-time conversion score.

Here is an example using curl with the live demo URL:
```
curl -X POST http://leadscore-demo.ezrahernowo.com/predict   
-H "Content-Type: application/json"   
-d '{  
      "TotalVisits": 10,  
      "Total Time Spent on Website": 600,  
      "Page Views Per Visit": 5,  
      "Lead Source": "Google",  
      "Last Activity": "Email Opened",  
      "Specialization": "Business Administration",  
      "What is your current occupation": "Working Professional"  
    }'
```

**Expected Response:**
```json
{  
  "lead_score": 85.3,  
  "prediction": "Will Convert"  
}
```

## **🔮 Future Work & Deployment Roadmap**

* **Advanced Feature Engineering:**  
  * [ ] Implement advanced feature selection techniques like **Recursive Feature Elimination (RFE)** to identify the most impactful features and potentially simplify the model.  
* **Model Optimization:**  
  * [ ] Perform **hyperparameter tuning** using GridSearchCV or RandomizedSearchCV to further optimize model performance.  
  * [ ] Experiment with more complex models like **Random Forest** or **Gradient Boosting (XGBoost)** to compare results.  
* **Interactive Dashboard:**  
  * [ ] Develop a dashboard using **Streamlit** or **Dash** that consumes the API, allowing users to input lead data via a web form and see the results visually.  
* **CI/CD Pipeline:**  
  * [ ] Implement a CI/CD pipeline using **GitHub Actions** to automatically test and deploy changes to the application.
