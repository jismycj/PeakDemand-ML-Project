============================================================
PeakDemand
Predicting Peak Electricity Demand Time Using Machine Learning
============================================================


TECHNOLOGY

Frontend:
HTML
CSS
JavaScript
Chart.js

Backend:
Python
Flask

Database:
MySQL

Machine Learning:
Pandas
NumPy
Scikit-learn
Random Forest Regression

Authentication:
Flask session
Werkzeug password hashing


============================================================
PROJECT FLOW
============================================================

Registration
      |
      v
Login
      |
      v
User / Admin Role
      |
      +-----------------------+
      |                       |
      v                       v
User Dashboard          Admin Dashboard
      |
      v
Upload CSV
      |
      v
Data Cleaning
      |
      v
Feature Extraction
      |
      +--------------------------+
      |                          |
      v                          v
Hour                     Day of Week
Month                    Season
Holiday                  Temperature
      |
      v
Random Forest
      |
      v
24 Hour Prediction
      |
      v
Peak Time + Peak Demand
      |
      v
Demand Level
      |
      v
Suggestion
      |
      v
Chart + Dashboard


============================================================
CSV COLUMNS
============================================================

Date_Time
Hour
Day
Month
Season
Holiday
Temperature
Demand_MW


============================================================
MYSQL TABLES
============================================================

users

datasets

dataset_rows

model_runs

predictions


============================================================
RUN
============================================================

1. Start MySQL.

2. Open VS Code terminal.

3. Activate virtual environment.

Windows:

venv\Scripts\activate

4. Install packages:

pip install -r requirements.txt

5. Create database:

python setup_database.py

6. Start application:

python run.py

7. Open browser:

http://127.0.0.1:5000


============================================================
ADMIN REGISTRATION
============================================================

Registration page contains:

User
Admin

For Admin registration use the key defined in .env:

PEAKADMIN2026

You can change this value.


============================================================
MODEL
============================================================

Random Forest Regression

Input:

Hour
Day
Month
Season
Holiday
Temperature

Target:

Demand_MW


============================================================
EVALUATION
============================================================

MAE
RMSE
R2


============================================================