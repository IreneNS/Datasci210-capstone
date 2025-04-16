# Infrastructure 
Dependencies and structure for a FastAPI application to host the benchmark and DSP models.

## Function
The function of this FastAPI application includes support for the GET endpoint `/health` for status purposes, as well as /benchmark and /model endpoints which can accept POST JSON payloads, and returns predicted portfolio weights from the models.

The `/benchmark` endpoint accepts a JSON payload from a POST. The payload is verified using a Pydantic model to ensure that it aligns with the expected format, including data types, ranges, and keys. If the input is accepted, it is run through a model which is intended to craft a table of portfolio weights based on the inputs of universe size and risk. If it is not accepted, an error is returned.

The `/model` endpoint accepts the same inputs as `/benchmark`, with similar input verification, and produces the same output, though with a deep learning model. In a use case, the inputs to `/benchmark` and `/model` are identical.

Predictions are stored in S3, with a record of existing inputs and matching predictions stored with DynamoDB.

There are also endpoints of `/docs` and `/openapi.json` which are accessible while the app is running, for the OpenAPI documentation and a JSON that meets the OpenAPI specification version 3+.

The application is deployed in an EC2 instance. The recommended minimum requirements for running this application are 2vCPU and 8 RAM. The endpoints are accessed from a Lambda function attached to API Gateway endpoints.

## Building
The application is built with FastAPI, using Poetry to manage dependencies. The application consists of a main application script, where all endpoints are located, and the models are stored in separate importable scripts `benchmark.py` and `dlmodel.py`. The application is then run with Poetry and Uvicorn from the instance. The IP of the instance is held constant with Elastic IP.

For dependencies, a Poetry project is initialized, adding the intended Python version, FastAPI standard, joblib, and scikit-learn as main dependencies, plus pytest and ruff as development dependencies.

Within the prediction endpoints, a pickled pre-trained model is loaded in. Pydantic is used for field validation of any inputs sent to the model. 

## Running
The application is run with uvicorn to launch the server, locally with the command `poetry run uvicorn src.main:app`. The endpoints can be reached with curl to check functionality, with properly formatted JSON inputs for the prediction endpoints.