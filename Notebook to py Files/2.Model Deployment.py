"""
# Model Deployment
"""

"""
## Packages
"""

# Importing the necessary libraries
import sys
import os
import io
import boto3
import sagemaker
from sagemaker.model import Model
from sagemaker.serializers import CSVSerializer
from sagemaker.deserializers import JSONDeserializer
from sagemaker.amazon.amazon_estimator import get_image_uri 
from sagemaker.session import s3_input, Session
from sagemaker import get_execution_role

import warnings
warnings.filterwarnings('ignore')

"""
## Step 1: Prepare Model Artifacts
"""

# Get the current original directory
original_directory = os.getcwd()

# Print curent directory
print("Original working directory: {0}".format(os.getcwd()))

# Change the current working directory to 'deployment_assets'
os.chdir('deployment_assets')

# Print the list of files and directories in the current working directory
print("List of files and directories in the {0} directory:".format(os.getcwd()))
display(os.listdir())

# Remove the existing 'model.tar.gz' file, if it exists
!rm -f model.tar.gz

# Remove the '.ipynb_checkpoints' directory recursively and forcefully
!rm -rf .ipynb_checkpoints

# Create a new tar.gz archive named 'model.tar.gz' containing all files and directories in the current directory
!tar czvf model.tar.gz "."

"""
## Step 2:  Upload Model Artifacts to S3
"""

# Specify the resource names
role = get_execution_role()
bucket = 'predicting-medical-appointment-no-shows' # Name of the S3 bucket
prefix = 'model' # The prefix (subdirectory path) in the S3 bucket
my_region = boto3.session.Session().region_name  # Get the current AWS region of this session

# Print the AWS bucket details and region
print("AWS Bucket: ", bucket)
print("Prefix (or Subdirectory): ", prefix)
print("AWS Region: ", my_region)

# Using Boto3 to create a connection
s3_client = boto3.client('s3')

# Retrieve the list of objects with the specified prefix in the bucket
contents = s3_client.list_objects(Bucket=bucket, Prefix=prefix)['Contents'] 
for f in contents:
    print(f['Key']) # Print each file key in the bucket's specified prefix

# Open the model.tar.gz file 
fObj = open("model.tar.gz", "rb")

# Define the S3 object key as a combination of prefix and file name
key = os.path.join(prefix, "model.tar.gz")
print(key)

# Upload the pre-trained model to S3
boto3.Session().resource("s3").Bucket(bucket).Object(key).upload_fileobj(fObj)

"""
## Step 3: Create SageMaker Model
"""

# Reset to the original directory
os.chdir(original_directory)

# Verify the reset
print("Current working directory reset to: {0}".format(os.getcwd()))

# create the sagemaker session
sess = sagemaker.Session()

# Specify the role
role = get_execution_role()

# Specify the location of the model in S3 (S3 URI)
model_url = 's3://predicting-medical-appointment-no-shows/model/model.tar.gz'

# Retrieve the Docker container image URI for the XGBoost model
container = sagemaker.image_uris.retrieve(region=my_region,
                                         framework='xgboost',
                                         version='1.7-1')

# Create a SageMaker Model (from custom model with a pre-built SageMaker container)
xgb_model = Model(model_data=model_url,
                  image_uri=container,
                  role=role,
                  entry_point='deployment_assets/inference.py',
                  name='medical-appointment-no-show-model',
                  sagemaker_session=sess)

"""
## Step 4: Deploy the Model to an Endpoint
"""

# Deploy the model
xgb_predictor = xgb_model.deploy(endpoint_name='medical-appointment-no-show-model-endpoint',
                                 initial_instance_count=1,
                                 instance_type='ml.m4.xlarge',
                                 serializer=CSVSerializer(),
                                 deserializer=JSONDeserializer())

"""
## Step 5: Create Lambda function
"""

"""
### Create new function
"""

"""
1. On the Lambda console, on the **Functions** page, choose **Create function**.
2. For **Function name**, enter a name (for example, `medical-appointment-no-show-lambda`).
3. For **Runtime**¸ choose a runtime.
4. For **Execution role**¸ select **Create a new role** or **Use an existing role**.
5. Click **Create function** button.

    <img src="Notebook_images/Create lambda function.png" alt="Image" style="width: 90%; height: 80%;" />
"""

"""
### Write function
"""

"""
1. Under **Code** tab, update ***lambda_function.py*** file with code below:
    ```python
    import json
    import os
    import boto3

    ENDPOINT_NAME = os.environ['ENDPOINT_NAME']
    runtime = boto3.client('runtime.sagemaker')

    def lambda_handler(event, context):
        try:
            print("Received event: " + json.dumps(event,indent=2))

            body = json.loads(event["body"])
            
            if "data" in body:
                payload = body["data"]
                print("Data:", payload)
            else:
                print("No 'data' key in the body")
                return {"error": "No 'data' key in the body"}

            # Invoke the SageMaker endpoint
            response = runtime.invoke_endpoint(EndpointName = ENDPOINT_NAME,
                                               Body = payload,
                                               ContentType = 'text/csv',
                                                   Accept = "application/json"
                                              )

            # Process the SageMaker response
            output = json.loads(response['Body'].read().decode("utf-8"))
            print(f"Predict output: {output}")

            print('ok')

            return {
                'statusCode': 200,
                'body': json.dumps(f"Predict output: {output}")
            }
            
        except Exception as e:
            print("Error occurred:", str(e))
            return {"error": str(e)}
            
    ```

2. Click the **Deploy** button.

    <img src="Notebook_images/Write lambda function.png" alt="Image" style="width: 90%; height: 80%;" />
"""

"""
### Add environment variable
"""

"""
1. Under **Configuration** tab, on the side panel, click **Environment variables**.
2. Click **Edit**.
    
    <img src="Notebook_images/Edit environment variable.png" alt="Image" style="width: 90%; height: 80%;" />

3. In the next page, click **Add Environment variable**.
4. Enter the **Key** with `ENDPOINT_NAME` and **Value** with `medical-appointment-no-show-model-endpoint`.
5. Click **Save**.

    <img src="Notebook_images/Add environment variable.png" alt="Image" style="width: 90%; height: 80%;" />
"""

"""
### Edit general configuration
"""

"""
1. Under **Configuration tab**, on the side panel, click **General configuration**.
2. Click **Edit**.

    <img src="Notebook_images/Edit general configuration.png" alt="Image" style="width: 90%; height: 80%;" />

3. In the next page, edit **Memory**, **Timeout** and other basic settings as needed.
4. Click **Save**.

    <img src="Notebook_images/Edit basic settings.png" alt="Image" style="width: 90%; height: 80%;" />
"""

"""
## Step 6: Create a REST API
"""

"""
### Create API
"""

"""
1. On the API Gateway console, choose the **REST API**.
2. Choose **Build**.

    <img src="Notebook_images/Create REST API.png" alt="Image" style="width: 90%; height: 80%;" />

3. Select **New API**.
4. For **API name**¸ enter a name (for example, `medical-appointment-no-show-API`).
5. Leave **Endpoint Type** as **Regional**.
6. Choose **Create API**.

    <img src="Notebook_images/Create API.png" alt="Image" style="width: 90%; height: 80%;" />
"""

"""
### Create method
"""

"""
1. In the **Resources** page, choose **Create method**.

    <img src="Notebook_images/Create method.png" alt="Image" style="width: 90%; height: 80%;" />

2. For **Method type**, choose **POST**.
3. For **Integration type**, select **Lambda Function**.
4. For **Lambda function**, choose the created function in step 4.
5. Toggle on **Lambda proxy integration**.
6. Click **Create method**.

    <img src="Notebook_images/Create POST method.png" alt="Image" style="width: 90%; height: 80%;" />
"""

"""
### Deploy API
"""

"""
1. In the **Resources** page, choose **Deploy API**.

    <img src="Notebook_images/Deploy API.png" alt="Image" style="width: 90%; height: 80%;" />

2. Create a new stage with stage name (for example, `V1`).
3. Choose **Deploy**.

    <img src="Notebook_images/API stage.png" alt="Image" style="width: 90%; height: 80%;" />
"""

"""
## Step 7: Test API from local client
"""

"""
### Copy API URL
"""

"""
1. In the created API, go to **Stages** page and select the deployed stage (for example, `V1`).
2. Copy the **Invoke URL**.

    <img src="Notebook_images/API URL.png" alt="Image" style="width: 90%; height: 80%;" />
"""

"""
### Convert test data in to JSON strinng
"""

##Cnnvert csv to json for testing with postman
import json
import pandas as pd

csv_file = "data/input/simulated_set.csv"

# Read the CSV file into a DataFrame
df = pd.read_csv(csv_file)

# Convert the DataFrame to a CSV string
csv_string = df.to_csv(index=False, header=True)

# Convert the CSV string to a JSON string
json_payload = json.dumps({"data": csv_string})

# Output the JSON string
print(json_payload)

"""
### Test API with Postmsan
"""

"""
1. Enter the copied invoke URL into Postman.
2. Choose **POST** as method.
3. On the **Body** tab, enter the converted JSON string from above cell block.
4. Choose **Send** to see the returned result.

    <img src="Notebook_images/Test API with POSTMAN.png" alt="Image" style="width: 90%; height: 80%;" />

5. Lastly, check if prediction has returned to S3.

    <img src="Notebook_images/S3 prediction.png" alt="Image" style="width: 90%; height: 80%;" />
"""

