# Train and save a ML model
The code generates a dummy dataset, and split it into train and test. It's a binary classification dataset where each data point is represented by 4 features.
Logistic Regression is the classification algorithm that is used to predict the probability of a categorical dependent variable.

# Use Flask to create an API to use the ML model
Created two endpoints in which a user can test the trained ML model. The first endpoint allows the user to test the ML model using default parameters, while the second endpoint allows the user to set the input parameters and get the prediction

# Dockerize the Flask application
Dockerize the Flask app to easily ship, test, and deploy the ML model in different computing environments
