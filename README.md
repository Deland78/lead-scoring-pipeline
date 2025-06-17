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
├── app-lead-scoring/                      # Contains the Flask Web APP
│   ├── app.py  
│   ├── docker-compose.yml 
│   ├── Dockerfile         
│   ├── requirements.txt    
│   └── gunicorn_config.py  
├── api-lead-scoring/                      # Contains the FAST API
│   ├── models/
│   ├── .dockerignore
│   ├── docker-compose.yml
│   ├── Dockerfile
│   ├── main.py     
│   ├── README.md 
│   └──requirements.txt    
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

1. **Clone the Repository:**
```
git clone https://github.com/1bytess/lead-scoring-pipeline.git  
cd lead-scoring-pipeline
```

2. **Create the Environment File:**  
The database service requires an .env file for credentials. Create a file named .env in the root directory and add the following:  
```
DB_USER=<user>
DB_PASSWORD=<your_database_password>
DB_HOST=<your_database_host> 
DB_PORT=5432  
DB_NAME=<your_database_name>
```
3. **Launch the app:**

- **Launch the UI APP Locally:**  
  From the root directory, run the following command to build and start the API service, the database, and all other components:  
  ```
  cd api-lead-scoring
  python3 app.py
  ```
  The API will be running and available at `http://localhost:5000`. *(Check your docker-compose.yml for the port you mapped for the app service).*

- **Launch the API on Docker:**  
  From the root directory, run the following command to build and start the API service, the database, and all other components:  
  ```
  cd api-lead-scoring
  docker-compose build --no-cache
  docker-compose up -d
  ```
  The API will be running and available at `http://<you-server-ip>:3001/v2`. *(Check your docker-compose.yml for the port you mapped for the app service).*

## **💡 How to Use the API**

You can send a POST request with lead data to the /predict endpoint to get a real-time conversion score.

Here is an example using curl with the live demo URL:
```
curl -X POST http://api.ezrahernowo.com/v2/predict \   
-H "Content-Type: application/json" \
-d '{  
      "TotalVisits": 4,  
      "Total Time Spent on Website": 1850,  
      "Page Views Per Visit": 4,
      "Lead Origin": "API"  
      "Lead Source": "Google",  
      "Last Activity": "SMS Sent",  
      "What is your current occupation": "Unemployed"  
    }'
```
or
```
curl -X POST https://api.ezrahernowo.com/v2/predict -H "Content-Type: application/json" -d '{"TotalVisits": 4, "Total Time Spent on Website": 1850, "Page Views Per Visit": 4, "Lead Origin": "API", "Lead Source": "Google", "Last Activity": "SMS Sent", "What is your current occupation": "Unemployed"}'
```
**Expected Response:**
```json
{
  "lable": "Will Conver",
  "lead_score": 88.89,  
  "prediction": 1
}
```

## **✨ Try the Live Dashboard App**
To see the project in action, visit the live web application:

[https://demo.ezrahernowo.com/lead-scoring](https://demo.ezrahernowo.com/lead-scoring)

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
