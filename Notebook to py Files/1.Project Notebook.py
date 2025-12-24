"""
# Predicting Medical Appointment No Shows

## Problem Statement
A person makes a doctor appointment, receives all the instructions and no-show. Who to blame?
Our client is a medical ERP solutions provider, who wants to address this no-show issue by building an ML model that will be able to predict whether a patient will be a show or no-show.
The objective is to build the model and incorporate it into their ERP software which will then be used by their customers to better manage their appointments. 

## Data Description
Below is a table that summarizes the data dictionary:

| No | Column Name      | Description                                                                                                           |
|----|------------------|-----------------------------------------------------------------------------------------------------------------------|
| 01 | PatientId        | Identification of a patient                                                                                           |
| 02 | AppointmentID    | Identification of each appointment                                                                                    |
| 03 | Gender           | Male or Female. Female is the greater proportion, women take way more care of their health in comparison to men.      |
| 04 | ScheduledDay     | The day someone called or registered the appointment, this is before the appointment of course.                       |
| 05 | AppointmentDay   | The day of the actual appointment, when they have to visit the doctor.                                                |
| 06 | Age              | How old is the patient.                                                                                               |
| 07 | Neighbourhood    | Where the appointment takes place.                                                                                    |
| 08 | Scholarship      | True or False. Indicates whether the patient is enrolled in Brasilian welfare program Bolsa Família.                  |
| 09 | Hypertension     | True or False. Indicates if the patient has hypertension.                                                             |
| 10 | Diabetes         | True or False. Indicates if the patient has diabetes.                                                                 |
| 11 | Alcoholism       | True or False. Indicates if the patient is an alcoholic.                                                              |
| 12 | Handcap          | True or False. Indicates if the patient is handicapped.                                                               |
| 13 | SMS_received     | True or False. Indicates if 1 or more messages sent to the patient.                                                   |
| 14 | No-show          | True or False (**Target variable**). Indicates if the patient missed their appointment.                               |

The dataset used for this project can be downloaded from [Kaggle](https://www.kaggle.com/datasets/joniarroba/noshowappointments).
"""

"""
## Modules Importation
Below are the modules required to run this notebook, some modules will need to be installed explicitly.
"""

%%capture

# Use src/requirements.txt to install necessary packages, instead of below individual pip install commands
# 1) conda create --name medical_app_env python=3.10 -y
# 2) cd src
# 3) pip install -r requirements.txt

# !pip install snowflake-connector-python
# !pip install snowflake-sqlalchemy
# !pip install category_encoders
# !pip install hyperopt
# !pip install imblearn
# !pip install xgboost

import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency, pointbiserialr

### SKLearn Imports ###
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, precision_recall_curve, auc, roc_auc_score, ConfusionMatrixDisplay,f1_score
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


from category_encoders import TargetEncoder
from xgboost import XGBClassifier
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

import sqlalchemy
import snowflake.connector
from sqlalchemy import create_engine
from snowflake.sqlalchemy import *
import src.snowflake_creds as snowflake_creds

import warnings
warnings.filterwarnings('ignore')

"""
## Data Loading from Snowflake
The downloaded dataset (Medical-Appointment-No-Show-Prediction\data\main\full-data\KaggleV2-May-2016.csv) from kaggle was first loaded to a Snowflake database for storage ([reference for loading data to Snowflake](https://www.chaosgenius.io/blog/snowflake-upload-csv/)). With utilizing SQL queries (Snowflake_assets\Create APPOINTMENT_DATA table.sql), a hypothesis based Exploratory Data Analysis was performed to test our assumptions (Snowflake_assets\Hypothesis based EDA.sql).
"""

"""
For this notebook, We're going to connect to the Snowflake database using [Snowflake SQLAlchemy](https://docs.snowflake.com/en/developer-guide/python-connector/sqlalchemy#:~:text=can%20use%20the-,snowflake.sqlalchemy.URL,-method%20to%20constructhttps://docs.snowflake.com/en/developer-guide/python-connector/sqlalchemy#:~:text=can%20use%20the-,snowflake.sqlalchemy.URL,-method%20to%20construct), execute a SQL query to retrieve the appointment data, and load the returned result into a Pandas DataFrame.
"""

# # Define the Snowflake connection parameters
# # Sagemaker
# '''
# Note: Account info can be obtained from
# full URL (https://{account info}.snowflakecomputing.com) of the Snowflake account
# '''
# engine = create_engine(URL(
#         account="lvb79593.us-east-1",
#         user= snowflake_creds.USER_NAME,
#         password= snowflake_creds.PASSWORD,
#         role="ACCOUNTADMIN",
#         warehouse="MY_WAREHOUSE",
#         database="MEDICAL_APPOINTMENT_NO_SHOW",
#         schema="APPOINTMENT_SCHEMA"
#     ))

# # Define the SQL query that retrieves the appointment data
# '''
# Note: Specify database and schema where table is stored under
# '''
# query = """
# SELECT * FROM MEDICAL_APPOINTMENT_NO_SHOW.APPOINTMENT_SCHEMA.APPOINTMENT_DATA;
# """

# # Use a context manager to ensure the connection is closed after executing the query
# try:
#     with engine.connect() as conn:
#         # Execute the query and load the result into a Pandas DataFrame
#         data = pd.DataFrame(pd.read_sql(query, conn))
#         # Convert column names to uppercase
#         data.columns = [col.upper() for col in data.columns.tolist()]
#         # Print connection successful message
#         print("Connection successful. Data loaded into DataFrame.")
        
# except Exception as e:  # Catch any exceptions that occur
#     print(f"An error occurred: {e}")

"""
![image.png](attachment:image.png)
"""

from dotenv import load_dotenv
load_dotenv()

import os
snowflake_user = os.getenv("SNOWFLAKE_USER")
snowflake_password = os.getenv("SNOWFLAKE_PASSWORD")
snowflake_account = os.getenv("SNOWFLAKE_ACCOUNT")
snowflake_role = os.getenv("SNOWFLAKE_ROLE")
snowflake_warehouse = os.getenv("SNOWFLAKE_WAREHOUSE")
snowflake_database = os.getenv("SNOWFLAKE_DATABASE")
snowflake_schema = os.getenv("SNOWFLAKE_SCHEMA")

# Define the Snowflake connection parameters
# Initial Code
'''
Note: Account info can be obtained from
full URL (https://{account info}.snowflakecomputing.com) of the Snowflake account
'''
engine = create_engine(URL(
        account=snowflake_account,
        user= snowflake_user,
        password= snowflake_password,
        role=snowflake_role,
        warehouse=snowflake_warehouse,
        database=snowflake_database,
        schema=snowflake_schema
    ))

# Define the SQL query that retrieves the appointment data
'''
Note: Specify database and schema where table is stored under
'''
query = """
SELECT * FROM MEDICAL_APPOINTMENT_NO_SHOW.APPOINTMENT_SCHEMA.APPOINTMENT_DATA;
"""

# Use a context manager to ensure the connection is closed after executing the query
try:
    with engine.connect() as conn:
        # Execute the query and load the result into a Pandas DataFrame
        data = pd.DataFrame(pd.read_sql(query, conn))
        # Convert column names to uppercase
        data.columns = [col.upper() for col in data.columns.tolist()]
        # Print connection successful message
        print("Connection successful. Data loaded into DataFrame.")
        
except Exception as e:  # Catch any exceptions that occur
    print(f"An error occurred: {e}")

"""
Optionally, we can save the data frame to our local disk for ease to load them back into our workspace for analysis. (Nobody train a ML model in one sit, they close the laptop and come back, so loading the data from snowflake is a time consuming task, so we save them to a pickle file and the loading is fast)
```python
# Save DataFrame to a pickle file
data.to_pickle("./data/input/full_data.pkl")
```
"""

"""
#### Sampling from the raw, unprocessed data to create a simulated set to mimic the real-world scenario where we receive new data without labels and need to make predictions   

#### We take a random sample from our original data and pretend it’s new, unseen data—just like what we’d get in real life when we want to make predictions. To make it realistic, we remove the answers (labels) from this sample, so our model doesn’t “cheat” by seeing the correct results. This helps us test how well our prediction process works on data where we don’t know the outcome yet.  

#### You are taking a portion of your original, unprocessed dataset and setting aside some rows as a "simulated set." In this set, you remove the target labels (set them to NaN), so it looks like new, unseen data you might get in the real world—where you don’t know the true outcomes yet. This allows you to test your prediction pipeline as if you were making predictions on real, unlabeled data.  

#### QN : Is the simulated set training data?
#### ANS: No, the simulated set is not training data. It is a separate sample taken from your original data, with the target labels removed, to mimic new, real-world data where you don’t know the outcomes. You use it to test your prediction pipeline, not to train your model.

#### Is this rows included in training dataset?
#### No, the rows in the simulated set are not included in the training dataset. They are separated out from the original data before training, so your model does not see them during training. This helps you test your model on truly unseen data, just like in a real-world prediction scenario.

```python
# Split the data to maintain the distribution of the target variable
_, simulated_set = train_test_split(
    data, 
    test_size=0.2, 
    stratify=data['NO_SHOW'],  # Stratify based on the target variable
    random_state=42
)

# Set the targets to NaN
simulated_set['NO_SHOW'] = np.nan

# Save the simulated set to a CSV file
simulated_set.to_csv('./data/input/simulated_set.csv', index=False)

print("Stratified simulated set saved to './data/input/simulated_set.csv'")
```
"""

# # Save the DataFrame as a pickle file
# # SageMaker
# data.to_pickle("./data/input/full_data.pkl")

## Save the DataFrame as a pickle file
## PC
# data.to_pickle("./data/input/full_data.pkl")

# Split the data to maintain the distribution of the target variable
_, simulated_set = train_test_split(
    data, 
    test_size=0.2, 
    stratify=data['NO_SHOW'],  # Stratify based on the target variable
    random_state=42
)

# Set the targets to NaN
simulated_set['NO_SHOW'] = np.nan

# Save the simulated set to a CSV file
simulated_set.to_csv('./data/input/simulated_set.csv', index=False)

print("Stratified simulated set saved to './data/input/simulated_set.csv'")

"""
## Data Inspection
We can first have a look at the apppointment dataset and inspect its basic attributes.
"""

# Read the saved pickle file as DataFrame
data = pd.read_pickle("./data/input/full_data.pkl")

# Check first 5 rows of data frame
data.head()

# Check shape of data frame
data.shape

# Check for null values in each column
data.isnull().sum()

# Check the data types of current columns
data.info()

# To see the unique value counts in gender column
data['GENDER'].value_counts()

data['NEIGHBOURHOOD'].value_counts()

data['NO_SHOW'].value_counts()

"""
**Observations**
1. *PATIENTID* is not needed as we're predicting the No-shows per appointment basis. *APPOINTMENTID* can be used as index as it represents each unique appointment.
2. *GENDER* and *NEIGHBOURHOOD* are both categorical features and can be numerically encoded.
3. *NO_SHOW*, our target variable, should also be numerically encoded so that it can be predicted via machine learning.
"""

"""
## Hypothesis-based Exploratory Data Analysis
As mentioned, with utilizing Snowflake SQL, a hypothesis based Exploratory Data Analysis was performed to test out our assumptions. The hypotheses are as below, for detailed work of the SQL queries, please refer to `Snowflake assets\Hypothesis based EDA.sql`.
"""

"""
### Hypothesis 1: Do Males tend to miss more appointments than Females? 

<details><summary> Click here for SQL query </summary>
    
``` SQL
SELECT 
    Gender,
    COUNT(*) AS Total_Appointments,
    SUM(CASE WHEN No_show = 'Yes' THEN 1 ELSE 0 END) AS Missed_Appointments,
    ROUND((SUM(CASE WHEN No_show = 'Yes' THEN 1 ELSE 0 END) / COUNT(*) * 100),1) AS Percentage_Missed
FROM 
    APPOINTMENT_DATA
GROUP BY 
    Gender;

``` 
</details>
<details><summary> Click here for Observations </summary> 
    
 - Both males and females have a similar rate of missing appointments, around 20%.

    <img src="Notebook_images/Hypothesis 1.png" alt="Image" style="width: 90%; height: 80%;" />
    
</details>
"""

"""
### Hypothesis 2: Is there relationship between scheduled-appointment day difference, and patient no-show?

<details><summary> Click here for SQL query </summary>

```SQL
WITH TimeDifference AS (
    SELECT 
        DATEDIFF(day, ScheduledDay, AppointmentDay) AS DaysDifference,
        No_show
    FROM 
        APPOINTMENT_DATA
)
SELECT 
    DaysDifference,
    COUNT(*) AS Total_Appointments,
    SUM(CASE WHEN No_show = 'Yes' THEN 1 ELSE 0 END) AS Missed_Appointments,
    ROUND((SUM(CASE WHEN No_show = 'Yes' THEN 1 ELSE 0 END) / COUNT(*) * 100),1) AS Percentage_Missed
FROM 
    TimeDifference
GROUP BY 
    DaysDifference
ORDER BY 
    DaysDifference;
```
</details>
<details><summary> Click here for Observations </summary>

- There are 5 appointments with negative days difference which they all have 100% missed appointment rate because they were scheduled for past date. This could be due to data entry errors.
- A significant number of appointments (38,563) are scheduled on the same day, with a relatively low missed appointment rate of 4.6%.

    <img src="Notebook_images/Hypothesis 2a.png" alt="Image" style="width: 90%; height: 80%;" />

- For time differences between 1 day to 80 days, the missed appointment rate fluctuates but generally stays within the range of around 20% to 40%.
- The longer time differences (>80 days) have varying missed appointment rates, like on:
    - day 83 (12.5%), 86 (16.7%)
    - day 103 (60%), day 104 (75%)
    - day 112, 115, 117, 119, 122 (0%) 
    - day 132, 139, 146, 151 (100%)

    <img src="Notebook_images/Hypothesis 2b.png" alt="Image" style="width: 90%; height: 90%;" />
</details>
"""

"""
### Hypothesis 3: Is the no-show common among adult patients aged between 18-30?

<details><summary> Click here for SQL query </summary>
    
```SQL
WITH AgeGroups AS (
    SELECT 
        CASE 
            WHEN Age BETWEEN 0 AND 12 THEN '0-12'
            WHEN Age BETWEEN 13 AND 17 THEN '13-17'
            WHEN Age BETWEEN 18 AND 30 THEN '18-30'
            WHEN Age BETWEEN 31 AND 50 THEN '31-50'
            ELSE '50+'
        END AS AgeGroup,
        No_show
    FROM 
        APPOINTMENT_DATA
)
SELECT 
    AgeGroup,
    COUNT(*) AS Total_Appointments,
    SUM(CASE WHEN No_show = 'Yes' THEN 1 ELSE 0 END) AS Missed_Appointments,
    ROUND((SUM(CASE WHEN No_show = 'Yes' THEN 1 ELSE 0 END) / COUNT(*) * 100),1) AS Percentage_Missed
FROM 
    AgeGroups
GROUP BY 
    AgeGroup
ORDER BY 
    AgeGroup;
```
</details>
<details><summary> Click here for Observations </summary>

- The age group of 13-17 years has the highest no-show rate at 26.6% (even they have least appointments scheduled), followed by age group of 18-30 with 24.6%  missed appointment rate.
- In contrast, the age group 50+ years has the lowest no-show rate at 16.2% (even they have highest number of appointments scheduled).

    <img src="Notebook_images/Hypothesis 3a.png" alt="Image" style="width: 90%; height: 80%;" />

    <img src="Notebook_images/Hypothesis 3b.png" alt="Image" style="width: 90%; height: 80%;" />

</details>
"""

"""
## Data Preprocessing
Based on the observations in the Data Inspection section, we will take the following preprocessing steps:
1. Drop *PATIENTID* column
2. Set *APPOINTMENTID* as index
3. Numerical encode target variable: *NO_SHOW*
4. Drop rows which the *APPOINTMENTDAY* comes before *SCHEDULEDDAY*. 
5. Drop rows with negative *AGE*.

The categorical input features *GENDER* and *NEIGHBOURHOOD* will be numerically encoded in feature engineering step.
"""

"""
### Drop redundant column(s)
We will drop the *PATIENTID* column as *APPOINTMENTID* is the more appropriate unique identifier for this dataset.
"""

# Define a list of column names to drop from the DataFrame
cols_to_drop = ['PATIENTID']

# Create a copy of the original DataFrame to avoid modifying the original data
data_pre = data.copy()

# Print the shape of the DataFrame before dropping columns
print("Before dropping: ", data_pre.shape)

# Drop interested columns and print the shape of the DataFrame after dropping columns
data_pre = data_pre.drop(cols_to_drop, axis=1) # axis = 1 indicates columns
print("After dropping: ", data_pre.shape)

data.head(3)

"""
### Reset index
The column APPOINTMENTID is the unique identifier to each individual appointment, therefore we can reset the Dataframe index to the APPOINTMENTID.
"""

# Print the total number of rows in the DataFrame
print(f"Rows in data frame: {data_pre.shape[0]}")

# Print the count of unique values in the 'APPOINTMENTID' column of the DataFrame
print(f"Unique value in APPOINTMENTID column: {data_pre['APPOINTMENTID'].nunique()}")

# YOU CAN SEE THAT PATIENTID HAS LESS UNIQUE VALUES THAN APPOINTMENTID
print(f"Unique value in PATIENTID column: {data['PATIENTID'].nunique()}")

# Set the 'APPOINTMENTID' column as the index of the DataFrame 'data_pre'
data_pre.set_index('APPOINTMENTID', inplace=True) # inplace=True to modify the DataFrame directly
# If we dont use

# Display the first 5 rows of the DataFrame
data_pre.head()

# Check shape of data frame
data_pre.shape

"""
### Numeric encode categorical target variable (No-Show)
The target variable *NO_SHOW* contains *yes/no* character Boolean, we can encode it to numeric Boolean of *1/0*.
"""

# Get the unique values in the 'NO_SHOW' column before encoding
data_pre['NO_SHOW'].unique()

# Encode the 'NO_SHOW' column, where 'No' becomes 0 and 'Yes' becomes 1
data_pre['NO_SHOW'] = data_pre['NO_SHOW'].map({'No': 0, 'Yes': 1})

# Display the unique values in the 'NO_SHOW' column to verify the encoding
data_pre['NO_SHOW'].unique()

"""
### Drop rows for appointment day comes before schedule day 
We can see ther are some records where the appointment day comes before schedule day, this could due to data entry error so we will drop these records.
"""

# Filter records where the calculated 'DAYS_TILL_APPOINTMENT' would be negative
negative_days_records = data_pre[(data_pre['APPOINTMENTDAY'] - data_pre['SCHEDULEDDAY']).dt.days + 1 < 0]

# Count the number of such records
count_negative_days = len(negative_days_records)

# Print the count and the records
print(f"Number of records with negative Days Till Appointment: {count_negative_days}")
negative_days_records

# Drop records where the calculated 'DAYS_TILL_APPOINTMENT' would be negative
data_pre = data_pre[(data_pre['APPOINTMENTDAY'] - data_pre['SCHEDULEDDAY']).dt.days + 1 >= 0]

# Check shape of data frame
data_pre.shape

"""
### Drop rows with negative age
There's 1 record with with age < 0, we can drop this row since it's not realistic to have age of -1.
"""

"""
Before we proceed, let's have a quick check and see if the age is within the valid range.
"""

# Display the minimum and maxiumum age 
print("Minimum age:", data_pre['AGE'].min())
print("Maximum age:", data_pre['AGE'].max())

"""
The minimum age of -1 does not make sense, let's have a look at how many records have age below 0.
"""

# Filter records where AGE is -1
records_with_negative_age = data_pre[data_pre['AGE'] < 0]

# Get the total number of records where AGE is -1
total_records_negative_age = len(records_with_negative_age)

# Display the records and total number of records with AGE = -1
print(f"Total number of records with AGE<0: {total_records_negative_age}")
records_with_negative_age

# Drop records where AGE is less than 0
data_pre = data_pre[data_pre['AGE'] >= 0]

# Check shape of data frame
data_pre.shape

# Display the minimum and maxiumum age 
print("Minimum age:", data_pre['AGE'].min())
print("Maximum age:", data_pre['AGE'].max())

"""
Now a minimum of 0 and maximum of 115 seem to be a valid range for age.
"""

"""
## Statistical Testing
In this section, we will apply Statistical Testing to help understand the impact of the features on the target. For example:
1. Perform Chi-square test between *Gender* and the target variable (*No-show*) to see if Gender impacts the target.
2. Perform Chi-square test between *Alcoholism* and the target variable (*No-show*) to see if Alcoholism impacts the target.
3. Is sending messages to the patients really useful? Check its significance with the target variable using relevant testing techniques.
"""

"""
### Chi-Square Test for Independence: Gender and No-Show
**Objective**<br>
To determine if there is a statistically significant relationship between the gender of the patient (*GENDER*) and the likelihood of missing an appointment (*NO_SHOW*).

**Null Hypothesis (H0)**<br>
The gender of the patient is independent of whether they will miss an appointment. In other words, gender does not impact the likelihood of a no-show.

**Alternative Hypothesis (H1)**<br>
The gender of the patient is not independent of whether they will miss an appointment. In other words, gender does impact the likelihood of a no-show.

**Significance Level**<br>
We will use a significance level ($\alpha$) of 0.05 for this test.

**Expected Outcome**<br>
If the p-value is less than the significance level, we will reject the null hypothesis, suggesting that gender does have an impact on the likelihood of a no-show. Otherwise, we will fail to reject the null hypothesis, suggesting that the two variables are independent.
"""

# Create the contingency table for Gender and NO_SHOW
contingency_table_gender = pd.crosstab(data_pre['GENDER'], data_pre['NO_SHOW'])

# Plotting the bar chart
fig, ax = plt.subplots()

# Set positions and width for the bars
pos = list(range(len(contingency_table_gender[1])))
bar_width = 0.35

# Plot bars for '1' and '0' in 'NO_SHOW'
plt.bar(pos, contingency_table_gender[1], bar_width, label='No-show = 1')
plt.bar([p + bar_width for p in pos], contingency_table_gender[0], bar_width, label='No-show = 0')

# Set axis labels and title
ax.set_xticks([p + 0.5 * bar_width for p in pos])
ax.set_xticklabels(['Female', 'Male'])
plt.xlabel('Gender')
plt.ylabel('Number of Appointments')
plt.title('Impact of Gender on No-shows')

# Add legend
plt.legend()

# Show the plot
plt.show()

# Create a contingency table
# This table will show the frequency distribution of no-shows across different genders
contingency_table = pd.crosstab(data_pre['GENDER'], data_pre['NO_SHOW'])

# Perform the Chi-square test
chi2, p_value, dof, expected = chi2_contingency(contingency_table)

# Print the results of the Chi-square test
print("Results")
print(f"Chi-square value: {chi2}") # The Chi-square statistic
print(f"P-value: {p_value}") # The p-value of the test
print(f"Degrees of Freedom: {dof}") # Degrees of freedom


# Interpret the results based on the p-value and a significance level of 0.05
alpha = 0.05  # Significance level
print("\nConclusion")
if p_value < alpha:
    print("Reject the null hypothesis. There is a significant relationship between Gender and No-show.")
else:
    print("Fail to reject the null hypothesis. There is no significant relationship between Gender and No-show.")

"""
### Chi-Square Test for Independence: Alcoholism and No-Show
**Objective**<br>
To determine if there is a statistically significant relationship between the alcoholism status of the patient (*Alcoholism*) and the likelihood of missing an appointment (*NO_SHOW*).

**Null Hypothesis (H0)**<br>
The alcoholism status of the patient is independent of whether they will miss an appointment. In other words, alcoholism does not impact the likelihood of a no-show.

**Alternative Hypothesis (H1)**<br>
The alcoholism status of the patient is not independent of whether they will miss an appointment. In other words, alcoholism does impact the likelihood of a no-show.

**Significance Level**<br>
We will use a significance level ($\alpha$) of 0.05 for this test.

**Expected Outcome**<br>
If the p-value is less than the significance level, we will reject the null hypothesis, suggesting that alcoholism does have an impact on the likelihood of a no-show. Otherwise, we will fail to reject the null hypothesis, suggesting that the two variables are independent.
"""

# Create the contingency table for Alcoholism and NO_SHOW
contingency_table_alcoholism = pd.crosstab(data_pre['ALCOHOLISM'], data_pre['NO_SHOW'])

# Plotting the bar chart
fig, ax = plt.subplots()

# Set positions and width for the bars
pos = list(range(len(contingency_table_alcoholism[1])))
bar_width = 0.35

# Plot bars for '1' and '0' in 'NO_SHOW'
plt.bar(pos, contingency_table_alcoholism[1], bar_width, label='No-show = 1')
plt.bar([p + bar_width for p in pos], contingency_table_alcoholism[0], bar_width, label='No-show = 0')

# Set axis labels and title
ax.set_xticks([p + 0.5 * bar_width for p in pos])
ax.set_xticklabels(['Not Alcoholic', 'Alcoholic'])
plt.xlabel('Alcoholism')
plt.ylabel('Number of Appointments')
plt.title('Impact of Alcoholism on No-shows')

# Add legend
plt.legend()

# Show the plot
plt.show()

# Create a contingency table
# This table will show the frequency distribution of no-shows across patients with and without alcoholism
contingency_table_alcoholism = pd.crosstab(data_pre['ALCOHOLISM'], data_pre['NO_SHOW'])

# Perform the Chi-square test
chi2_alcoholism, p_value_alcoholism, dof_alcoholism, expected_alcoholism = chi2_contingency(contingency_table_alcoholism)

# Print the results of the Chi-square test
print("Results")
print(f"Chi-square value: {chi2_alcoholism}")  # The Chi-square statistic
print(f"P-value: {p_value_alcoholism}")  # The p-value of the test
print(f"Degrees of Freedom: {dof_alcoholism}")  # Degrees of freedom

# Interpret the results based on the p-value and a significance level of 0.05
alpha = 0.05  # Significance level
print("\nConclusion")
if p_value_alcoholism < alpha:
    print("Reject the null hypothesis. There is a significant relationship between Alcoholism and No-show.")
else:
    print("Fail to reject the null hypothesis. There is no significant relationship between Alcoholism and No-show.")

"""
### Chi-Square Test for Independence: SMS Received and No-Show
**Objective**<br>
To determine if there is a statistically significant relationship between receiving an SMS (*SMS_received*) and the likelihood of missing an appointment (*NO_SHOW*).

**Null Hypothesis (H0)**<br>
Receiving an SMS is independent of whether a patient will miss an appointment. In other words, receiving an SMS does not impact the likelihood of a no-show.

**Alternative Hypothesis (H1)**<br>
Receiving an SMS is not independent of whether a patient will miss an appointment. In other words, receiving an SMS does impact the likelihood of a no-show.

**Significance Level**<br>
We will use a significance level ($\alpha$) of 0.05 for this test.

**Expected Outcome**<br>
If the p-value is less than the significance level, we will reject the null hypothesis, suggesting that receiving an SMS does have an impact on the likelihood of a no-show. Otherwise, we will fail to reject the null hypothesis, suggesting that the two variables are independent.
"""

# Create the contingency table
contingency_table_sms = pd.crosstab(data_pre['SMS_RECEIVED'], data_pre['NO_SHOW'])

# Plotting the bar chart
fig, ax = plt.subplots()

# Set the positions and width for the bars
pos = list(range(len(contingency_table_sms[1])))  # Using 1 for no-show
bar_width = 0.35

# Plot bars for '1' and '0' in 'NO_SHOW'
plt.bar(pos, contingency_table_sms[1], bar_width, label='No-show = 1')
plt.bar([p + bar_width for p in pos], contingency_table_sms[0], bar_width, label='No-show = 0')

# Set axis labels and title
ax.set_xticks([p + 0.5 * bar_width for p in pos])
ax.set_xticklabels(['Not Received SMS', 'Received SMS'])
plt.xlabel('SMS Received')
plt.ylabel('Number of Appointments')
plt.title('Impact of SMS on No-shows')

# Add legend
plt.legend()

# Show the plot
plt.show()

# Create a contingency table
# This table will show the frequency distribution of no-shows for patients who received an SMS and those who didn't
contingency_table_sms = pd.crosstab(data_pre['SMS_RECEIVED'], data_pre['NO_SHOW'])

# Perform the Chi-square test
chi2_sms, p_value_sms, dof_sms, expected_sms = chi2_contingency(contingency_table_sms)

# Print the results of the Chi-square test
print("Results")
print(f"Chi-square value: {chi2_sms}")  # The Chi-square statistic
print(f"P-value: {p_value_sms}")  # The p-value of the test
print(f"Degrees of Freedom: {dof_sms}")  # Degrees of freedom

# Interpret the results based on the p-value and a significance level of 0.05
alpha = 0.05  # Significance level
print("\nConclusion")
if p_value_sms < alpha:
    print("Reject the null hypothesis. There is a significant relationship between SMS_received and No-show.")
else:
    print("Fail to reject the null hypothesis. There is no significant relationship between SMS_received and No-show.")

"""
##  Stratified Hold-out Split
In this section, we will be performing a stratified hold-out split on the dataset. The goal is to divide the data into training and testing sets while maintaining the distribution of the target variable, *NO_SHOW*, in both subsets. This ensures that the model will be trained and evaluated on data that is representative of the overall distribution of the target variable.

We will also visualize the distribution of *NO_SHOW* before and after the split to confirm that the stratification worked as expected.
"""

# Count the frequency of each unique value in the 'NO_SHOW' column
data_pre['NO_SHOW'].value_counts()

# Visualize the distribution of 'NO_SHOW' before splitting using a pie chart
plt.figure(figsize=(6, 6))
data_pre['NO_SHOW'].value_counts().plot(kind='pie', autopct='%1.1f%%', startangle=90, labels=['Show', 'No Show'])
plt.title('NO-SHOW Distribution Before Splitting')
plt.ylabel('')  # Hide the y-axis label for clarity
plt.show()

"""
We can now perform the 80-20 stratified split and verify the target class distribution stays the same before and after splitting.
"""

# Perform the 80%-20% stratified split
train_data, test_data = train_test_split(data_pre, test_size=0.2, stratify=data_pre['NO_SHOW'], random_state=42)

# Calculate the value counts for the 'NO_SHOW' column in the training set
train_value_counts = train_data['NO_SHOW'].value_counts(normalize=True) * 100

# Calculate the value counts for the 'NO_SHOW' column in the test set
test_value_counts = test_data['NO_SHOW'].value_counts(normalize=True) * 100

# Round the value counts to 1 decimal place
train_value_counts = train_value_counts.round(1)
test_value_counts = test_value_counts.round(1)

# Display the rounded value counts
print("Training set 'NO_SHOW' value counts (in %):")
display(train_value_counts)

print("\nTest set 'NO_SHOW' value counts (in %):")
display(test_value_counts)

# Check the shape of the training set
print(f"Shape of training set: {train_data.shape}")

# Check the shape of the testing set
print(f"Shape of testing set: {test_data.shape}")

"""
## Feature Engineering and Additional Data Preprocessing
Creating new features or selecting important features from the existing dataset that can help improve the model's performance. This may involve techniques like feature extraction, dimensionality reduction, or creating interaction variables.
"""

"""
### Create new features
Combining multiple features into a single feature can sometimes reveal interesting patterns that are not apparent when considering the features individually. For example:
1. Analyze how *Gender* and *Age* together impacts the target variable and create a new feature which takes in Gender and Age buckets.
2. Using *Scheduled_Day* and *Appointment_Day*, create a new feature that says days till appointment for each patient.
"""

"""
#### GENDER_AGE 
One of the key features we will create is *GENDER_AGE*, a combined feature of *GENDER* and *AGE_BUCKET*. This new feature aims to capture the interaction between a patient's gender and age group, which could have a significant impact on whether they will show up for their medical appointment.
"""

"""
##### Combine gender and bucketized age
we will first create a temporary column called *AGE_BUCKET*, which bucketizes *Age* into age groups, then combine *Gender* and the *AGE_BUCKET* as *GENDER_AGE*, and finally drop the temporary *AGE_BUCKET* column.
"""

bins = [0, 13, 18, 31, 51, np.inf]
labels = ['0-12', '13-17', '18-30', '31-50', '50+']

# Create AGE_BUCKET for training set and then create GENDER_AGE
train_data['AGE_BUCKET'] = pd.cut(train_data['AGE'], bins=bins, labels=labels, right=False)
train_data['GENDER_AGE'] = train_data['GENDER'] + '_' + train_data['AGE_BUCKET'].astype(str)

# Reorder columns to make 'NO_SHOW' the last column in train_data
cols_train = list(train_data.columns)
cols_train.remove('NO_SHOW')
cols_train.append('NO_SHOW')
train_data = train_data[cols_train]

# Create AGE_BUCKET for testing set and then create GENDER_AGE
test_data['AGE_BUCKET'] = pd.cut(test_data['AGE'], bins=bins, labels=labels, right=False)
test_data['GENDER_AGE'] = test_data['GENDER'] + '_' + test_data['AGE_BUCKET'].astype(str)

# Reorder columns to make 'NO_SHOW' the last column in test_data
cols_test = list(test_data.columns)
cols_test.remove('NO_SHOW')
cols_test.append('NO_SHOW')
test_data = test_data[cols_test]

# Drop AGE_BUCKET from training set
train_data.drop('AGE_BUCKET', axis=1, inplace=True)

# Drop AGE_BUCKET from testing set
test_data.drop('AGE_BUCKET', axis=1, inplace=True)

# Check the shape of the training set
print(f"Shape of training set: {train_data.shape}")

# Check the shape of the testing set
print(f"Shape of testing set: {test_data.shape}")

"""
##### Chi-Square Test: GENDER_AGE and No-Show
**Objective**<br>
To understand if the combined feature of gender and age group has a significant impact on the likelihood of a patient missing their appointment.

**Null Hypothesis (H0)**<br>
There is no significant relationship between *GENDER_AGE* and *NO_SHOW*.

**Alternative Hypothesis (H1)**<br>
There is a significant relationship between *GENDER_AGE* and *NO_SHOW*.

**Significance Level**<br>
We will use a significance level ($\alpha$) of 0.05 for this test.
    
**Expected Outcome**<br>
If the p-value obtained from the Chi-Square Test is less than the significance level, we will reject the null hypothesis, indicating that *GENDER_AGE* has a significant impact on *NO_SHOW*. Otherwise, we fail to reject the null hypothesis.
"""

"""
We will only perform the test on the training set. This is because the training set is what we'll be using to build our model, and we want to understand the relationships within that specific subset of the data. 

Additionally, keeping the test set untouched ensures its integrity for model evaluation later on.
"""

# Create the contingency table for GENDER_AGE and NO_SHOW in the training set
contingency_table_gender_age_train = pd.crosstab(train_data['GENDER_AGE'], train_data['NO_SHOW'])

# Perform the Chi-square test
chi2, p_value, dof, expected = chi2_contingency(contingency_table_gender_age_train)

# Print the results
print("Results")
print(f"Chi-square value: {chi2}")
print(f"P-value: {p_value}")
print(f"Degrees of Freedom: {dof}")

# Interpret the results
alpha = 0.05  # Significance level
print("\nConclusion")
if p_value < alpha:
    print("Reject the null hypothesis. There is a significant relationship between GENDER_AGE and No-show in the training set.")
else:
    print("Fail to reject the null hypothesis. There is no significant relationship between GENDER_AGE and No-show in the training set.")

"""
#### DAYS_TILL_APPOINTMENT
In this section, we introduce a new feature called *DAYS_TILL_APPOINTMENT*, which represents the number of days between the scheduled day and the actual appointment day. This feature could potentially help us understand if the time gap between scheduling and the appointment has any impact on the likelihood of a patient missing their appointment.
"""

"""
##### Subtract scheduled day from appointment day
Create a new feature *DAYS_TILL_APPOINTMENT* by subtracting scheduled day from appointment day.
"""

# Calculate the number of days till the appointment for both training sets
train_data['DAYS_TILL_APPOINTMENT'] = (train_data['APPOINTMENTDAY'] - train_data['SCHEDULEDDAY']).dt.days + 1

# Reorder columns to make 'NO_SHOW' the last column in train_data
cols_train = list(train_data.columns)
cols_train.remove('NO_SHOW')
cols_train.append('NO_SHOW')
train_data = train_data[cols_train]

# Calculate the number of days till the appointment for test sets
test_data['DAYS_TILL_APPOINTMENT'] = (test_data['APPOINTMENTDAY'] - test_data['SCHEDULEDDAY']).dt.days + 1

# Reorder columns to make 'NO_SHOW' the last column in train_data
cols_train = list(train_data.columns)
cols_train.remove('NO_SHOW')
cols_train.append('NO_SHOW')
train_data = train_data[cols_train]

"""
Let's verify there's no negative days till appointment and check out the shapes of all available datasets to confirm our operations.
"""

# Check for negative values in the 'DAYS_TILL_APPOINTMENT' column
negative_days = train_data[train_data['DAYS_TILL_APPOINTMENT'] < 0]

# Print out the number of rows with negative days
print(f"Number of rows with negative 'DAYS_TILL_APPOINTMENT': {len(negative_days)}")

# Check the shape of the training set
print(f"Shape of training set: {train_data.shape}")

# Check the shape of the testing set
print(f"Shape of testing set: {test_data.shape}")

"""
##### Point-Biserial Correlation: DAYS_TILL_APPOINTMENT and No-Show
**Objective**<br>
To evaluate the impact of the feature *DAYS_TILL_APPOINTMENT* on the target variable *NO_SHOW*.

**Null Hypothesis (H0)**<br>
There is no relationship between *DAYS_TILL_APPOINTMENT* and *NO_SHOW*.

**Alternative Hypothesis (H1)**<br>
There is a relationship between *DAYS_TILL_APPOINTMENT* and *NO_SHOW*.

**Significance Level ($\alpha$)**<br>
0.05

**Expected Outcome**<br>
If the p-value is less than the significance level, we reject the null hypothesis.
"""

# Calculate the Point-Biserial Correlation Coefficient
result = pointbiserialr(train_data['DAYS_TILL_APPOINTMENT'], train_data['NO_SHOW'])

# Print the results
print("Point-Biserial Correlation Coefficient:", result.correlation)
print("P-value:", result.pvalue)

# Interpret the results
alpha = 0.05  # Significance level
print("\nConclusion")
if result.pvalue < alpha:
    print(f"Reject the null hypothesis. There is a significant relationship between DAYS_TILL_APPOINTMENT and NO_SHOW with a correlation coefficient of {result.correlation}.")
else:
    print("Fail to reject the null hypothesis. There is no significant relationship between DAYS_TILL_APPOINTMENT and NO_SHOW.")

"""
**Observation**

A Point-Biserial Correlation Coefficient of 0.184 suggests a low but positive correlation between *DAYS_TILL_APPOINTMENT* and *NO_SHOW*. This means that as the number of days until the appointment increases, the likelihood of a no-show also slightly increases. However, the relationship is not strong.
"""

"""
### Preprocess dateetime columns
After creating the two new features, we continue to implement additional preprocessing steps. We focus on two key tasks: first, the extraction of meaningful features from datetime columns, and second, the removal of features that do not provide valuable information for our models.
"""

"""
#### Extract year, month, and day from SCHEDULEDDAY and APPOINTMENTDAY
In our dataset, we have two datetime columns: *SCHEDULEDDAY* and *APPOINTMENTDAY*. Machine learning models require numerical or categorical data, so we need to convert these datetime columns into a format that the model can understand. We'll extract the year, month, and day as separate features for each of these columns.
"""

# List of all datasets
datasets = [train_data, test_data]

# Loop through each dataset to perform the datetime feature extraction
for dataset in datasets:
    # Extracting year, month, and day from SCHEDULEDDAY
    dataset['SCHEDULEDDAY_YEAR'] = dataset['SCHEDULEDDAY'].dt.year
    dataset['SCHEDULEDDAY_MONTH'] = dataset['SCHEDULEDDAY'].dt.month
    dataset['SCHEDULEDDAY_DAY'] = dataset['SCHEDULEDDAY'].dt.day

    # Extracting year, month, and day from APPOINTMENTDAY
    dataset['APPOINTMENTDAY_YEAR'] = dataset['APPOINTMENTDAY'].dt.year
    dataset['APPOINTMENTDAY_MONTH'] = dataset['APPOINTMENTDAY'].dt.month
    dataset['APPOINTMENTDAY_DAY'] = dataset['APPOINTMENTDAY'].dt.day

    # Dropping the original datetime columns
    dataset.drop(['SCHEDULEDDAY', 'APPOINTMENTDAY'], axis=1, inplace=True)

# Reorder columns to make 'NO_SHOW' the last column in train_data
cols_train = list(train_data.columns)
cols_train.remove('NO_SHOW')
cols_train.append('NO_SHOW')
train_data = train_data[cols_train]

# Reorder columns to make 'NO_SHOW' the last column in test_data
cols_test = list(test_data.columns)
cols_test.remove('NO_SHOW')
cols_test.append('NO_SHOW')
test_data = test_data[cols_test]

"""
Let's check out the shapes of all available datasets to confirm our operations.
"""

# Check the shape of the training set
print(f"Shape of training set: {train_data.shape}")

# Check the shape of the testing set
print(f"Shape of testing set: {test_data.shape}")

"""
Before extracting we have 14 columns, we added 6 columns of schedule day's and appointment day's of day, month, year then drop the orginal *SCHEDULEDDAY* and *APPOINTMENTDAY* therefore 14+6-2=18.
"""

"""
#### Delete year column with just one unique value
If all the years are the same in a particular column, then that feature likely won't provide any useful information for the model. In that case, we can safely drop it.
"""

# List of all datasets
datasets = [train_data, test_data]

# Columns to check for unique values
year_columns_to_check = ['SCHEDULEDDAY_YEAR', 'APPOINTMENTDAY_YEAR']

# Loop through each dataset
for dataset in datasets:
    for col in year_columns_to_check:
        # If the column has only one unique value, drop it
        if dataset[col].nunique() == 1:
            print(f"Dropping {col} as it has only one unique value.")
            dataset.drop([col], axis=1, inplace=True)

"""
Let's check out the shapes of all available datasets to confirm our operations.
"""

# Check the shape of the training set
print(f"Shape of training set: {train_data.shape}")

# Check the shape of the testing set
print(f"Shape of testing set: {test_data.shape}")

"""
We dropped *APPOINTMENTDAY_YEAR*.
"""

"""
### Numeric encode categorical features
In this section, we will encode the categorical features *GENDER*, *NEIGHBOURHOOD*, and *GENDER_AGE* to prepare them for machine learning algorithms. Different encoding techniques will be applied based on the nature and distribution of each categorical variable.
"""

"""
Let's first verify our which columns are categorical and their corresponding total unique values.
"""

# Identify categorical columns in the training set
categorical_columns = train_data.select_dtypes(include=['object']).columns.tolist()

# Print the categorical columns
print("Categorical Columns:", categorical_columns)

# Check the number of unique values in each categorical column in training set
print("Training data unique values count for Categorical Columns:")
for col in ['GENDER', 'NEIGHBOURHOOD', 'GENDER_AGE']:
    print(f"{col} has {train_data[col].nunique()} unique values.")

# Check the number of unique values in each categorical column in test set
print("Test data unique values count for Categorical Columns:")
for col in ['GENDER', 'NEIGHBOURHOOD', 'GENDER_AGE']:
    print(f"{col} has {test_data[col].nunique()} unique values.")

"""
#### Encode GENDER
Since *GENDER* is a binary categorical feature, we will map 'Male' to 1 and 'Female' to 0. This is a straightforward method that avoids adding extra dimensions to our dataset.
"""

# Encoding the 'GENDER' column for the original training set
train_data['GENDER'] = train_data['GENDER'].map({'M': 1, 'F': 0})

# Encoding the 'GENDER' column for the test set
test_data['GENDER'] = test_data['GENDER'].map({'M': 1, 'F': 0})

"""
Let's check out the unique values for *GENDER* to confirm our operations.
"""

# Check the unique values in the 'GENDER' column for the training set
unique_values_train = train_data['GENDER'].unique()
print(f"Unique values in 'GENDER' column for training set: {unique_values_train}")

# Check the unique values in the 'GENDER' column for the test set
unique_values_test = test_data['GENDER'].unique()
print(f"Unique values in 'GENDER' column for test set: {unique_values_test}")

"""
#### Encode GENDER_AGE
For *GENDER_AGE*, we will use one-hot encoding. This feature has a moderate number of unique values, making one-hot encoding a suitable choice.
"""

# One-hot encode 'GENDER_AGE' for all datasets
train_data = pd.get_dummies(train_data, columns=['GENDER_AGE'], drop_first=True, dtype=int)
test_data = pd.get_dummies(test_data, columns=['GENDER_AGE'], drop_first=True, dtype=int)

# Reorder columns to make 'NO_SHOW' the last column in train_data
cols_train = list(train_data.columns)
cols_train.remove('NO_SHOW')
cols_train.append('NO_SHOW')
train_data = train_data[cols_train]

# Reorder columns to make 'NO_SHOW' the last column in test_data
cols_test = list(test_data.columns)
cols_test.remove('NO_SHOW')
cols_test.append('NO_SHOW')
test_data = test_data[cols_test]

"""
Let's have a look at the data frames to confirm our operations.
"""

# Check the first few rows of the training set to see the new columns
print("First few rows of training data:")
display(train_data.head())

# Check the first few rows of the test set to see the new columns
print("\nFirst few rows of test data:")
display(test_data.head())

# Check if all data types of the new columns in the training set are integers
print("\nAre all new columns in training data of integer type?")
print(all(train_data.filter(like='GENDER_AGE').dtypes == 'int'))

# Check if all data types of the new columns in the test set are integers
print("\nAre all new columns in test data of integer type?")
print(all(test_data.filter(like='GENDER_AGE').dtypes == 'int'))

# Check that both datasets have the same columns in the same order after encoding
print("\nDo both datasets have the same columns in the same order after encoding?")
print(list(train_data.columns) == list(test_data.columns))

"""
**Observations**
- Before encoding: 17 columns
- GENDER_AGE (One-Hot Encoded but with `drop_first=True`): +9 columns

17 originals columns + 9 new columns = 26 total columns.<br>
Since the original *GENDER_AGE* column will be dropped, therefore we have 26  - 1 = 25 total columns.
"""

"""
#### Encode NEIGHBOURHOOD
*NEIGHBOURHOOD* has a high cardinality. To handle this, we will use target encoding. This method will help us capture the relationship between the neighborhood and the target variable, *NO_SHOW*.
"""

# Target encode 'NEIGHBOURHOOD' for all datasets
encoder = TargetEncoder()
train_data['NEIGHBOURHOOD'] = encoder.fit_transform(train_data['NEIGHBOURHOOD'], train_data['NO_SHOW'])
test_data['NEIGHBOURHOOD'] = encoder.transform(test_data['NEIGHBOURHOOD'])

"""
Let's check for missing values, unique values, and data types of all available datasets to confirm our operations.
"""

# Check the data type of the 'NEIGHBOURHOOD' column in both datasets
print("Data type in training data:", train_data['NEIGHBOURHOOD'].dtype)
print("Data type in test data:", test_data['NEIGHBOURHOOD'].dtype)

# Check for missing values in the 'NEIGHBOURHOOD' column in both datasets
print("Missing values in training data:", train_data['NEIGHBOURHOOD'].isnull().sum())
print("Missing values in test data:", test_data['NEIGHBOURHOOD'].isnull().sum())

# Check the number of unique values in the 'NEIGHBOURHOOD' column in both datasets
print("Unique values in training data:", train_data['NEIGHBOURHOOD'].nunique())
print("Unique values in test data:", test_data['NEIGHBOURHOOD'].nunique())

"""
**Observations**
1. **Data Types**: Both the training and test datasets have the *NEIGHBOURHOOD* column as `float64`, which is a numerical data type. This is expected after target encoding.
  
2. **Missing Values**: There are no missing values in the *NEIGHBOURHOOD* column for both datasets, which is a good sign.

3. **Unique Values**: The number of unique values in the training data is 81 and in the test data is 78. This is expected behavior, as target encoding will map each unique category to a different numerical value based on the mean of the target variable for that category.
"""

"""
After all the encodings are done, there should be no categorical columns left.
"""

# Check if there are any non-numeric columns in each dataset
print("Number of non-numeric columns in the original training set:", len(train_data.select_dtypes(exclude=['number']).columns))
print("Number of non-numeric columns in the test set:", len(test_data.select_dtypes(exclude=['number']).columns))

"""
### Feature Correlation Analysis
In this section, we analyze the correlations between different features to identify any significant multicollinearity which could impact the performance of certain predictive models (such as logistic regression). A correlation matrix is computed and visualized as a heatmap for an intuitive understanding of these relationships. We particularly focus on identifying very high correlations (coefficients > 0.8 or < -0.8) as indicators of multicollinearity. The outcomes of this analysis will inform subsequent feature selection and modeling decisions.
"""

# Calculate the correlation matrix for the training data
corr_matrix = train_data.corr()

# Create a heatmap to visualize the correlation matrix
plt.figure(figsize=(15, 12))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.show()

"""
**Observation**

There doesn't appear to be any multicollinearity (very high correlation like > 0.8 or < -0.8) among the variables.
"""

"""
### Tackle Class Imbalance of No-Show
Class imbalance can significantly impact the performance of machine learning models. We will first check if there's an imbalance between the two classes in the target variable *NO_SHOW*. Then, to address this, we will employ three different sampling techniques:
1. Upsampling (Oversampling)
2. Downsampling (Undersampling)
3. Synthetic Minority Over-sampling Technique (SMOTE)

We only resample the training set. The test set should remain untouched to serve as an unbiased evaluation metric for the model. Resampling only the training set ensures that the model generalizes well to new, unseen data.
"""

"""
Before we proceed, let's first check the distribution of the target variable in the training set.
"""

# Calculate the actual counts of the 'NO_SHOW' column
no_show_counts = train_data['NO_SHOW'].value_counts().reset_index()
no_show_counts.columns = ['NO_SHOW', 'COUNT']

# Calculate the distribution of the 'NO_SHOW' column in percentages (rounded to 1 decimal place)
no_show_percentage = (train_data['NO_SHOW'].value_counts(normalize=True) * 100).round(1).reset_index()
no_show_percentage.columns = ['NO_SHOW', 'PERCENTAGE']

# Merge the counts and percentages into a single DataFrame
no_show_distribution = pd.merge(no_show_counts, no_show_percentage, on='NO_SHOW')

# Display the distribution in a table
print(no_show_distribution)

"""
#### Upsampling
Upsampling involves randomly duplicating observations from the minority class to balance the dataset.
"""

# Separate majority and minority classes in the original training set
train_majority = train_data[train_data.NO_SHOW == 0]
train_minority = train_data[train_data.NO_SHOW == 1]

# Upsample minority class
train_minority_upsampled = resample(train_minority, 
                                    replace=True,     # sample with replacement
                                    n_samples=len(train_majority),    # to match majority class
                                    random_state=123) # reproducible results

# Combine majority class with upsampled minority class
train_upsampled = pd.concat([train_majority, train_minority_upsampled]).reset_index(drop=True)

"""
#### Downsampling
Downsampling involves randomly removing observations from the majority class to balance the dataset.
"""

# Downsample majority class
train_majority_downsampled = resample(train_majority, 
                                      replace=False,    # sample without replacement
                                      n_samples=len(train_minority),  # to match minority class
                                      random_state=123) # reproducible results

# Combine minority class with downsampled majority class
train_downsampled = pd.concat([train_majority_downsampled, train_minority]).reset_index(drop=True)

"""
#### SMOTE
SMOTE creates synthetic samples from the minority class by interpolating between existing observations, enhancing diversity and aiding in balancing the dataset.
"""

# Separate features and target variable from original training data
X_train = train_data.drop('NO_SHOW', axis=1)
y_train = train_data['NO_SHOW']

# Initialize SMOTE object
smote = SMOTE(random_state=42)

# Fit SMOTE and obtain a balanced dataset
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Combine the resampled features and labels into a single DataFrame
train_smote = pd.concat([pd.DataFrame(X_train_smote), pd.DataFrame(y_train_smote, columns=['NO_SHOW'])], axis=1)

"""
Let's check out the distribution of all available sampling datasets to confirm our operations.
"""

print("Class distribution in upsampled data:", train_upsampled['NO_SHOW'].value_counts())
print("\nClass distribution in downsampled data:", train_downsampled['NO_SHOW'].value_counts())
print("\nClass distribution in SMOTE data:", train_smote['NO_SHOW'].value_counts())

"""
All three sampling techniques have successfully balanced the class distribution in our training data. Now each class has an equal number of instances.
"""

"""
### Feature Scaling
In this section, we will perform feature scaling to standardize the range of our independent variables. Scaling is crucial for algorithms that use distance-based metrics.
"""

"""
#### Apply StandardScaler
We will use Standard Scaling to transform the data into a distribution with a mean of 0 and a standard deviation of 1.
"""

# Initialize the scaler
scaler = StandardScaler()

# List of feature columns to scale
feature_cols = train_data.columns.difference(['NO_SHOW'])

# Create copies of the original data
train_data_scaled = train_data.copy()
test_data_scaled = test_data.copy()

# Apply scaling to the feature columns in the training and test sets
train_data_scaled[feature_cols] = scaler.fit_transform(train_data[feature_cols])
test_data_scaled[feature_cols] = scaler.transform(test_data[feature_cols])

# Create copies of the resampled data
train_upsampled_scaled = train_upsampled.copy()
train_downsampled_scaled = train_downsampled.copy()
train_smote_scaled = train_smote.copy()

# Apply scaling to the feature columns in the resampled datasets
train_upsampled_scaled[feature_cols] = scaler.transform(train_upsampled[feature_cols])
train_downsampled_scaled[feature_cols] = scaler.transform(train_downsampled[feature_cols])
train_smote_scaled[feature_cols] = scaler.transform(train_smote[feature_cols])

"""
Check if we have implemented Standard Scaling correctly.
"""

# Define a function to check if the mean is close to 0 and the standard deviation is close to 1
def check_scaling(df, cols_to_scale):
    return all(np.isclose(df[col].mean(), 0, atol=2e-1) and np.isclose(df[col].std(), 1, atol=2e-1) for col in cols_to_scale)

# List of feature columns to scale
feature_cols = train_data.columns.difference(['NO_SHOW'])

# Perform the sanity check on all datasets
print("Sanity check for original scaled training set:", check_scaling(train_data_scaled, feature_cols))
print("Sanity check for scaled test set:", check_scaling(test_data_scaled, feature_cols))
print("Sanity check for scaled upsampled set:", check_scaling(train_upsampled_scaled, feature_cols))
print("Sanity check for scaled downsampled set:", check_scaling(train_downsampled_scaled, feature_cols))
print("Sanity check for scaled SMOTE set:", check_scaling(train_smote_scaled, feature_cols))

"""
### Feature Selection
In this section, we will perform feature selection to identify the most important features for our model. Feature selection is crucial for making the model simpler, faster, and more interpretable.

To further refine our feature selection, we will use machine learning models to compute the feature importance. We will use Logistic Regression and Decision Trees for this purpose. This step will help us understand which features are most influential in predicting the target variable, *NO_SHOW*.
"""

"""
#### Logistic Regression Coefficient
We will look at the coefficients of the logistic regression model to understand the impact of each feature on the target variable.

We will focus on features that have at least 1% importance in either of the models.
"""

# Initialize the Logistic Regression model
logistic_model = LogisticRegression(class_weight='balanced', random_state=42)

# Fit the model on the scaled training data
logistic_model.fit(train_data_scaled.drop('NO_SHOW', axis=1), train_data_scaled['NO_SHOW'])

# Get the coefficients from the trained model
logistic_coef = pd.DataFrame(logistic_model.coef_.reshape(-1, 1), index=train_data_scaled.drop('NO_SHOW', axis=1).columns, columns=['Coefficient'])

# Plotting the feature importance
plt.figure(figsize=(15, 10))
logistic_coef['Coefficient'].sort_values().plot(kind='barh')
plt.title('Feature Importance based on Logistic Regression Coefficients')
plt.xlabel('Coefficient Value')
plt.ylabel('Features')
plt.show()

# Filter features with at least 1% importance (0.01)
important_features_logistic = logistic_coef[abs(logistic_coef['Coefficient']) >= 0.01]

# Display the important features and their count
print("Important features based on Logistic Regression:")
print(important_features_logistic)
print(f"Number of important features in Logistic Regression: {len(important_features_logistic)}")

"""
#### Decision Tree Feature Importance
We will use a Decision Tree model to get the feature importance scores.

We will focus on features that have at least 1% importance in either of the models.
"""

# Initialize the Decision Tree model
decision_tree_model = DecisionTreeClassifier(class_weight='balanced', random_state=42)

# Fit the model on the training data
decision_tree_model.fit(train_data.drop('NO_SHOW', axis=1), train_data['NO_SHOW'])

# Get the feature importances
feature_importances = decision_tree_model.feature_importances_

# Create a DataFrame for the feature importances
feature_importances_df = pd.DataFrame({'Feature': train_data.drop('NO_SHOW', axis=1).columns, 'Importance': feature_importances})

# Sort the DataFrame by the importances
feature_importances_df = feature_importances_df.sort_values('Importance', ascending=False)

# Filter features with at least 1% importance
important_features_tree = feature_importances_df[feature_importances_df['Importance'] >= 0.01]

# Plotting all feature importances
plt.figure(figsize=(15, 8))
plt.barh(feature_importances_df['Feature'], feature_importances_df['Importance'])
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importances from Decision Tree')
plt.show()

print("Important features based on Decision Tree:")
print(important_features_tree)
print(f"Number of important features in Decision Tree: {len(important_features_tree)}")

"""
#### Combine important features from linear and tree-based models
We will combine the important features based on the union of one linear (logistic regression) and one tree based (decision tree) model.
"""

# Convert the important features to lists
important_features_logistic_list = important_features_logistic.index.values.tolist()
important_features_tree_list = important_features_tree['Feature'].values.tolist()

# Find the intersection of both lists
combined_important_features_union = list(set(important_features_logistic_list) | 
                                                set(important_features_tree_list))

print("Combined important features (Union):")
display(combined_important_features_union)
print(f"Total number of combined important features (Intersection): {len(combined_important_features_union)}")

# Add 'NO_SHOW' to the list of important features
combined_important_features_union.append('NO_SHOW')

print(f"Total number of combined important features (Union) with target: {len(combined_important_features_union)}")

"""
#### Filter data sets with combined important features 
We will filter all our available datasets with the combined important features and target.
"""

# Filter the original training and test sets
train_data_filtered = train_data[combined_important_features_union]
test_data_filtered = test_data[combined_important_features_union]

# Filter the scaled training and test sets
train_data_scaled_filtered = train_data_scaled[combined_important_features_union]
test_data_scaled_filtered = test_data_scaled[combined_important_features_union]

# Filter the upsampled, downsampled, and SMOTE sets
train_upsampled_filtered = train_upsampled[combined_important_features_union]
train_downsampled_filtered = train_downsampled[combined_important_features_union]
train_smote_filtered = train_smote[combined_important_features_union]

# Filter the scaled and upsampled, downsampled, and SMOTE sets
train_upsampled_scaled_filtered = train_upsampled_scaled[combined_important_features_union]
train_downsampled_scaled_filtered = train_downsampled_scaled[combined_important_features_union]
train_smote_scaled_filtered = train_smote_scaled[combined_important_features_union]

"""
## Modeling
"""

"""
### Dataset Preperation
We will prepare the datasets that will be used for training and testing the models.
"""

# Original training sets
X_train = train_data_filtered.drop('NO_SHOW', axis=1)
y_train = train_data_filtered['NO_SHOW']

# Original test sets
X_test = test_data_filtered.drop('NO_SHOW', axis=1)
y_test = test_data_filtered['NO_SHOW']

# Scaled training and test sets
X_train_scaled = train_data_scaled_filtered.drop('NO_SHOW', axis=1)
X_test_scaled = test_data_scaled_filtered.drop('NO_SHOW', axis=1)

# Upsampled set
X_train_upsampled = train_upsampled_filtered.drop('NO_SHOW', axis=1)
y_train_upsampled = train_upsampled_filtered['NO_SHOW']

# Downsampled set
X_train_downsampled = train_downsampled_filtered.drop('NO_SHOW', axis=1)
y_train_downsampled = train_downsampled_filtered['NO_SHOW']

# SMOTE set
X_train_smote = train_smote_filtered.drop('NO_SHOW', axis=1)
y_train_smote = train_smote_filtered['NO_SHOW']

# Scaled and upsampled set
X_train_upsampled_scaled = train_upsampled_scaled_filtered.drop('NO_SHOW', axis=1)

# Scaled and downsampled set
X_train_downsampled_scaled = train_downsampled_scaled_filtered.drop('NO_SHOW', axis=1)

# Scaled and SMOTE set
X_train_smote_scaled = train_smote_scaled_filtered.drop('NO_SHOW', axis=1)

"""
### Model Selection
We will train multiple machine learning models (Logistic Regression, Decision Tree, Random Forest, and XGBoost) with Stratified K-Fold cross-validation on the original dataset to identify the one that performs best based on the ROC AUC.
"""

# Initialize Stratified K-Fold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Initialize models
logistic_model = LogisticRegression(class_weight='balanced', random_state=42)
decision_tree_model = DecisionTreeClassifier(class_weight='balanced', random_state=42)
random_forest_model = RandomForestClassifier(class_weight='balanced', random_state=42)
xgb_model = XGBClassifier(scale_pos_weight=(0.798 / 0.202), random_state=42)

# Perform cross-validation for each model
models = {'Logistic Regression': logistic_model, 
          'Decision Tree': decision_tree_model, 
          'Random Forest': random_forest_model, 
          'XGBoost': xgb_model}

for name, model in models.items():
    if name == 'Logistic Regression':
        score = cross_val_score(model, X_train_scaled, y_train, cv=skf, scoring='roc_auc', n_jobs=-1).mean()
    else:
        score = cross_val_score(model, X_train, y_train, cv=skf, scoring='roc_auc', n_jobs=-1).mean()
    
    print(f"Results for {name}:")
    print(f"ROC AUC: {score}")
    print("------")

"""
XGBoost is the best performing model with highest ROC AUC trained on the original dataset.
"""

"""
### Dataset selection
We will fit the best model (XGBoost) on priginal data set as well as different resampled data sets, and evaluate on the test set. This is to see how the model performs on different versions of the dataset. It will help us identify which resampling technique (if any) improves the model's performance on the test set.
"""

# Initialize the XGBoost model for original dataset
xgb_model_original = XGBClassifier(scale_pos_weight=(0.798 / 0.202), random_state=42)

# Initialize the XGBoost model for resampled datasets
xgb_model_resampled = XGBClassifier(random_state=42)

# Datasets
datasets = {
    'Original': (X_train, y_train),
    'Upsampled': (X_train_upsampled, y_train_upsampled),
    'Downsampled': (X_train_downsampled, y_train_downsampled),
    'SMOTE': (X_train_smote, y_train_smote)
}

# To store results
results = {}

# Loop through each dataset
for name, (X, y) in datasets.items():
    if name == 'Original':
        model = xgb_model_original
    else:
        model = xgb_model_resampled
    
    # Fit the model
    model.fit(X, y)
    
    # Make predictions on the test set
    y_pred = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1]  # Get the probabilities for the positive class
    
    # Evaluate the model
    roc_auc = roc_auc_score(y_test, y_pred_prob)  # Use probabilities for ROC AUC
    f1 = f1_score(y_test, y_pred)
    
    results[name] = {'ROC AUC': roc_auc, 'F1 Score': f1}

# Display the results
for name, metrics in results.items():
    print(f"Results for {name}:")
    print(f"ROC AUC: {metrics['ROC AUC']}")
    print(f"F1 Score: {metrics['F1 Score']}")
    print("------")

"""
Based on the results, the XGBoost model trained on the original dataset and evaluated on test set performs slightly better in terms of both ROC AUC and F1 Score compared to the models trained on the resampled datasets.
"""

"""
### Revisit Feature Importance
After we've selected the best model and dataset combination. We can revisit feature importance to help understand which features are driving the model's decisions and possibly simplify the model by removing less important features.
"""

# Retrain the model on the Upsampled dataset
xgb_model_original = XGBClassifier(scale_pos_weight=(0.798 / 0.202), random_state=42)
xgb_model_original.fit(X_train, y_train)

# Extract feature importances from the trained XGBoost model
feature_importances = xgb_model_original.feature_importances_

# Create a DataFrame for feature importances
feature_importance_df = pd.DataFrame({'Feature': X_train.columns, 'Importance': feature_importances})

# Sort the DataFrame by the importances
feature_importance_df = feature_importance_df.sort_values('Importance', ascending=False)

# Plot the feature importances
plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
plt.title('Feature Importance - XGBoost (Original)')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()

# Filter features with at least 1% importance
important_features_revisited = feature_importance_df[feature_importance_df['Importance'] >= 0.01]

print("Important features based on XGBoost (Original):")
print(important_features_revisited)
print(f"Number of important features: {len(important_features_revisited)}")

# Filter the training and test sets to include only the new set of important features
new_important_features = important_features_revisited['Feature'].tolist()
X_train_filtered = X_train[new_important_features]
X_test_filtered = X_test[new_important_features]

# Refit the XGBoost model on the filtered downsampled training set
xgb_model_original = XGBClassifier(scale_pos_weight=(0.798 / 0.202), random_state=42)
xgb_model_original.fit(X_train_filtered, y_train)

# Make prediction on the test set
y_pred_refit = xgb_model_original.predict(X_test_filtered)
y_pred_prob_refit = xgb_model_original.predict_proba(X_test_filtered)[:, 1]

# Evaluate the refitted model on the test set
roc_auc_refit = roc_auc_score(y_test, y_pred_prob_refit)
f1_score_refit = f1_score(y_test, y_pred_refit)
print(f"Results for XGBoost (Original) with New Important Features:")
print(f"ROC AUC: {roc_auc_refit}")
print(f"F1 Score: {f1_score_refit}")

"""
With the same threshold of 0.01, we have further reduced feature space to 21 important features , without losing the model's predictive power.
"""

"""
#### Save reduced input features
We will save the new features without target into pickle. This allows us to easily reuse them in future runs of our analysis or in deployment scenarios, ensuring consistency and reproducibility.
"""

# Save the new important features to a pickle file
with open('./data/features/X_train_important_features.pkl', 'wb') as f:
    pickle.dump(new_important_features, f)

"""
### Hyperparameter Tuning
We will fine-tune the hyperparameters of the selected model using [Hyperopt](https://hyperopt.github.io/hyperopt/#algorithms). The objective is to maximize the ROC AUC score. Hyperopt employs Bayesian optimization to find the best hyperparameters more efficiently than grid search or random search.

Here's a general outline of the steps we will follow:
1. Define the Objective Function: This function will take in hyperparameters, train the Gradient Boosting model, and return the metric we want to optimize (in this case, ROC AUC).
2. Define the Hyperparameter Space: Specify the range of values for each hyperparameter we want to tune.
    - **`n_estimators`**: The number of boosting rounds or trees to build. It's important to tune it properly as a very large number would make the model overfit. Range: [50, 1000].
    - **`max_depth`**: The maximum depth of the individual estimators. The depth of the tree can be used to control over-fitting. Range: [3, 14].
    - **`learning_rate`**: Step size shrinkage used to prevent overfitting. Range is [0.01, 0.2].
    - **`gamma`**: Minimum loss reduction required to make a further partition on a leaf node of the tree. The larger, the more conservative the algorithm will be. Range: [0, 0.5].
    - **`colsample_bytree`**: The fraction of features to choose for each boosting round. Used for subsampling of columns. Range: [0.3, 1].
    - **`subsample`**: The fraction of samples to be used for each boosting round. Range: [0.6, 1].
    - **`min_child_weight`**: Minimum sum of instance weight (hessian) needed in a child. Used to control over-fitting. Range: [1, 10].
3. Run Optimization: Use Hyperopt to run the optimization process.
"""

# Load the important features from the pickle file
with open('./data/features/X_train_important_features.pkl', 'rb') as f:
    important_features = pickle.load(f)

# Filter the original X_train and X_test dataset to include only the important features
X_train_filtered = X_train[important_features]
X_test_filtered = X_test[important_features]

best_score = 0  # Initialize the best score
iteration = 0  # Initialize the iteration counter

# Define the objective function
def objective(params):
    global best_score  # Declare best_score as global to update it
    global iteration  # Declare iteration as global to update it
    iteration += 1  # Increment the iteration counter
    
    clf = XGBClassifier(
        n_estimators=int(params['n_estimators']),
        max_depth=int(params['max_depth']),
        learning_rate=params['learning_rate'],
        gamma=params['gamma'],
        colsample_bytree=params['colsample_bytree'],
        subsample=params['subsample'],
        min_child_weight=int(params['min_child_weight']),
        scale_pos_weight=(0.798 / 0.202),
        random_state=42
    )
    
    # Stratified K-Fold cross-validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    score = cross_val_score(clf, X_train_filtered, y_train, cv=skf, scoring='roc_auc', n_jobs=-1).mean()
    
    if score > best_score:
        best_score = score
        print(f"New best score at iteration {iteration}: {best_score}")
        display("Best parameters so far:", params)
    
    return {'loss': -score, 'status': STATUS_OK}

# Define the parameter space
space = {
    'n_estimators': hp.quniform('n_estimators', 50, 1000, 1),
    'max_depth': hp.quniform('max_depth', 3, 14, 1),
    'learning_rate': hp.uniform('learning_rate', 0.01, 0.2),
    'gamma': hp.uniform('gamma', 0, 0.5),
    'colsample_bytree': hp.uniform('colsample_bytree', 0.3, 1),
    'subsample': hp.uniform('subsample', 0.6, 1),
    'min_child_weight': hp.quniform('min_child_weight', 1, 10, 1)
}

# Initialize a trials object
trials = Trials()

# Run the hyperparameter optimization
best = fmin(
    fn=objective,
    space=space,
    algo=tpe.suggest,
    max_evals=100,
    trials=trials
)

"""
The best loss (which is actually the negative of the ROC AUC score, because Hyperopt minimizes the loss) is approximately 0.744. This suggests that hyperparameter tuning has led to a slight improvement in the model's performance as compared to the mean ROC AUC score we obtained before tuning of 0.734.
"""

"""
#### Save best hyperparameters
We will save the best hyperparameters to a pickle file, which can be loaded later for training the final model.
"""

# Best parameters obtained from hyperparameter tuning
best_params = {
    'n_estimators': int(344),  # Convert float to int
    'max_depth': int(11),  # Convert float to int
    'learning_rate': 0.01024371856856813,
    'gamma': 0.4565024075152735,
    'colsample_bytree': 0.5547103400612381,
    'subsample': 0.8528421424055362,
    'min_child_weight': int(5),  # Convert float to int
    'scale_pos_weight': 0.798 / 0.202,
    'random_state': 42
}

# Save the best hyperparameters to a pickle file
with open('./data/hyperparameters/XGBoost_hyperparameters.pkl', 'wb') as f:
    pickle.dump(best_params, f)
    
print("Best hyperparameters saved to './data/hyperparameters/XGBoost_hyperparameters.pkl'")

"""
### Train Final Model
After hyperparameter tuning and model selection, the final model chosen was a Gradient Boosting Classifier trained on downsampled data. The model's performance is also evaluated on the test data, and the results are as follows.
"""

# Load the best hyperparameters
with open('./data/hyperparameters/XGBoost_hyperparameters.pkl', 'rb') as f:
    best_params = pickle.load(f)

# Initialize the XGBoost model
xgb_model_final = XGBClassifier(**best_params)

# Fit the model on the filtered training set
xgb_model_final.fit(X_train_filtered, y_train)

# Make predictions on the test set
y_pred_final = xgb_model_final.predict(X_test_filtered)
y_pred_prob = xgb_model_final.predict_proba(X_test_filtered)[:, 1]

# Evaluate the model
roc_auc_final = roc_auc_score(y_test, y_pred_prob)
f1_score_final = f1_score(y_test, y_pred_final)

print(f"Final Model Evaluation:")
print(f"ROC AUC: {roc_auc_final}")
print(f"F1 Score: {f1_score_final}")

"""
The ROC AUC score has improved from 0.7309 to 0.7372 while the F1 Score has also improved from 0.4418 to 0.4495. Although the improvement is not drastic, it's still a positive change that suggests the model is better at distinguishing between the positive and negative classes.
"""

"""
#### Save final model
We will save the final trained model to a pickle file. This will allow us to easily load the model later for making predictions or further analysis.
"""

# Save the trained model to a pickle file
with open("./model/xgb_model.pkl", "wb") as f:
    pickle.dump(xgb_model_final, f)

print("Model has been saved as './model/xgb_model.pkl'")

"""
### Final Model Evaluation
This section presents a thorough evaluation of our predictive model for forecasting patient no-shows at medical appointments. Key metrics and visualizations are used to assess accuracy and reliability:
- **Confusion Matrix**: Shows the model's correct and incorrect predictions, highlighting its effectiveness in classifying no-shows and shows.
- **Classification Report**: Provides precision, recall, and F1-score for each class, indicating the model's accuracy and balance in prediction.
- **ROC Curve**: Evaluates the model's ability to distinguish between no-show and show cases, as indicated by the Area Under the Curve (AUC).
- **Precision-Recall Curve**: Especially relevant for our imbalanced dataset, this curve illustrates the trade-off between precision and recall, reflected in the AUC-PR score.
"""

# Load the trained model from the pickle file
with open("./model/xgb_model.pkl", "rb") as f:
    loaded_model = pickle.load(f)

# Make predictions on the test set using the loaded model
y_pred_final = loaded_model.predict(X_test_filtered)
y_pred_prob = loaded_model.predict_proba(X_test_filtered)[:, 1]

# Compute confusion matrix
cm = confusion_matrix(y_test, y_pred_final)

# Display the confusion matrix
ConfusionMatrixDisplay(cm, display_labels=['No-Show', 'Show']).plot()
plt.title('Confusion Matrix')
plt.show()

"""
Based on the confusion matrix above, with label "No-Show" is coded as 0 and "Show" is coded as 1, the matrix can be interpreted as follows:
- True Negative (TN): 10,711 - The number of actual "No-Show" cases that were correctly predicted as "No-Show".
- False Positive (FP): 6,931 - The number of actual "No-Show" cases that were incorrectly predicted as "Show".
- False Negative (FN): 1,162 - The number of actual "Show" cases that were incorrectly predicted as "No-Show".
- True Positive (TP): 3,301 - The number of actual "Show" cases that were correctly predicted as "Show".
"""

# Generate classification report
report = classification_report(y_test, y_pred_final, target_names=['No-Show', 'Show'])
print("Classification Report:")
print(report)

"""
The model has high precision but lower recall for the "No-Show" class, meaning it's good at correctly identifying "No-Shows" but misses a significant number of them. For the "Show" class, the model has high recall but low precision, meaning it identifies most of the "Shows" but also has a lot of false positives.

The F1-Score for "No-Show" is decent at 0.73, but it's lower for "Show" at 0.45, indicating room for improvement, especially in balancing precision and recall for the "Show" class.

Overall, the model has an accuracy of 0.63, which represents the ratio of correct predictions to the total number of instances.
"""

# Compute the ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

# Compute the area under the ROC curve
roc_auc = auc(fpr, tpr)

# Plot the ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=1, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.show()

"""
The ROC curve plots the True Positive Rate (TPR) against the False Positive Rate (FPR) at various threshold settings. The area under the curve (AUC) quantifies the overall ability of the model to discriminate between the positive and negative classes.

The model has an AUC of 0.73, which means it has moderate ability to distinguish between the positive and negative classes. The trade-off involved is that in order to achieve a high recall (True Positive Rate), the model will also yield a high False Positive Rate. 

Ideally, we want a model that has a high recall (it captures most of the positive instances) and a low Positive Rate(it doesn't flag many negative instances as positive). This would correspond to an AUC-ROC close to 1.
"""

# Compute the precision-recall curve
precision, recall, _ = precision_recall_curve(y_test, y_pred_prob)

# Compute the area under the precision-recall curve
pr_auc = auc(recall, precision)

# Plot the precision-recall curve
plt.figure()
plt.plot(recall, precision, color='darkorange', lw=1, label=f'PR curve (area = {pr_auc:.2f})')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc='lower right')
plt.show()

"""
The PR curve plots the trade-off between precision and recall at various threshold settings. In the context of imbalanced datasets, the AUC-PR can provide a more informative picture of model performance than AUC-ROC, especially when the positive class (minority class) is of greater interest. In an ideal scenario, we would want the AUC-PR to be closer to 1, which would mean that the model has high precision and high recall. 

The area under the Precision-Recall curve (AUC-PR) of 0.38 indicates that  that the model struggles to achieve a good balance between precision and recall. In another words, The model is quite good at identifying most of the actual no-shows (high recall), but it also incorrectly flags many patients who would have shown up as no-shows (lower precision).
"""

"""
## Areas for Improvement and Recommendations
Based on the evaluations above, below are some areas for improvement and recommendations:
"""

"""
### Areas for Improvement
1. **ROC AUC Score**: While our ROC AUC score of 0.73 indicates moderate ability to distinguish between the positive and negative classes, it could be improved for a more reliable model.

2. **F1 Score**: The F1 score is a balance between precision and recall, and our model's F1 score of 0.45 suggests that it could do better, especially for the minority class.

3. **Precision-Recall Curve (AUC-PR)**: An AUC-PR of 0.38 indicates that the model's performance in terms of both precision and recall for the minority class could be significantly improved.

4. **Classification Report**: The report shows that while the model has a high recall for the minority class, the precision is low. This suggests that the model is identifying too many false positives.

5. **Confusion Matrix**: The number of false positives and false negatives could be reduced for a more balanced model.
"""

"""
### Recommendations
1. **Enhanced Feature Engineering**: To further improve our model's ROC AUC, F1 score, and AUC-PR, we recommend adding two new features based on the patient's appointment history:
    - **Total Prior Appointments**: This feature tracks the total number of appointments a patient has had before the current appointment. It provides a broader context of the patient's engagement with healthcare services.
    - **Total Missed Appointments**: This feature counts the number of appointments the patient has missed in the past. It directly relates to the likelihood of future no-shows, as a pattern of missing appointments could indicate a higher probability of not showing up.
    
    
    By incorporating these historical data points, the model can better understand each patient's appointment attendance patterns, leading to more accurate predictions. These features have been shown to **significantly enhance** model performance in similar studies ([reference](https://www.linkedin.com/pulse/predict-medical-show-appointments-wael-dagash-/)), addressing the areas of ROC AUC, F1 score, and AUC-PR, which are crucial for a balanced and effective predictive model.

2. **Advanced Resampling Techniques**: Try more advanced resampling techniques like ADASYN or Borderline-SMOTE to balance the classes.

3. **Ensemble Methods**: Use ensemble methods like stacking or boosting with different base models to improve performance.

4. **Model Selection**: Explore other algorithms that are well-suited for imbalanced datasets, such as LightGBM or CatBoost.

5. **Threshold Tuning**: Adjust the classification threshold based on the precision-recall trade-off that is most suitable for our specific problem.

6. **Hyperparameter Tuning**: Revisit hyperparameter tuning with a broader or different range of parameters, especially after making other changes like feature engineering.
"""

