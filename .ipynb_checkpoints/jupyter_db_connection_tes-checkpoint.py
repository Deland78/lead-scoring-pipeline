import pandas as pd
from sqlalchemy import create_engine
import os
from dotenv import load_dotenv
from pathlib import Path

# --- Database Connection Details ---
db_user = os.getenv('DB_USER')
db_password = os.getenv('DB_PASSWORD')
db_host = os.getenv('DB_HOST')
db_port = os.getenv('DB_PORT')
db_name = os.getenv('DB_NAME')

# Create the database connection URL
env_path = Path('.') / '.env'
load_dotenv(dotenv_path=env_path)

# --- Database Connection Details (Loaded from .env file) ---
db_user = os.getenv('DB_USER')
db_password = os.getenv('DB_PASSWORD')
db_host = os.getenv('DB_HOST')
db_port = os.getenv('DB_PORT')
db_name = os.getenv('DB_NAME')

# Create the database connection URL
db_url = f"postgresql+psycopg2://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"

print("Connecting to the database...")

# Check if credentials were loaded correctly
if not all([db_user, db_password, db_host, db_name]):
    print("Error: Database credentials not found. Make sure your .env file is correct and saved.")
else:
    try:
        # Create a SQLAlchemy engine
        engine = create_engine(db_url)

        # Define the SQL query to fetch all data from the 'leads' table
        query = "SELECT * FROM leads;"

        # Use pandas to read the data from the database into a DataFrame
        leads_df = pd.read_sql(query, engine)

        print("Successfully loaded data into a pandas DataFrame!")
        
        # --- Verify the Data ---
        print("\nFirst 5 rows of the lead scoring data:")
        print(leads_df.head())

        print(f"\nDataFrame shape: {leads_df.shape}")

    except Exception as e:
        print(f"An error occurred: {e}")