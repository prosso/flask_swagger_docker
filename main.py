import pandas as pd
import numpy as np
from flask import Flask, jsonify, request
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
import pickle
import os
from flasgger import Swagger
import flasgger

"""
-delete all images: docker system prune -a
-list all images: docker images
-build new image: docker build -t flask-tutorial:latest .
-run new image: docker run -p 5000:5000 flask-tutorial
-run the software: http://127.0.0.1:5000/apidocs

-save the docker image: docker save flask-tutorial > flask-tutorial.tar
-delete all images: docker system prune -a
-load docker image: docker load --input flask-tutorial.tar
-rnu docker image: docker run -p 5000:5000 flask-tutorial
-run the software: http://127.0.0.1:5000/apidocs
"""


def train_and_save_model():

    # generate dummy data
    X,y = make_classification(n_samples=1000, n_features=4, n_classes=2)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42,stratify=y) # 20% stratified test split

    logistic_regression = LogisticRegression(random_state=42)
    logistic_regression.fit(X_train,y_train)
    predictions = logistic_regression.predict(X_test)

    print("Accuracy Score of Model:", str(accuracy_score(y_test, predictions)))
    print("Classification Report:\n", str(classification_report(y_test,predictions)))

    file_name = 'LogisticRegression.pkl'
    output = open(file_name, 'wb')
    pickle.dump(logistic_regression, output)
    output.close()

# train_and_save_model()
app = Flask(__name__)
Swagger(app)

@app.route('/predict', methods = ['GET'])
def get_predictions():

    """
    API that returns the predicted class given the four features as input parameter
    ---

    parameters:
       - name: feature_1
         in: query
         type: number
         required: true

       - name: feature_2
         in: query
         type: number
         required: true

       - name: feature_3
         in: query
         type: number
         required: true

       - name: feature_4
         in: query
         type: number
         required: true

    responses:
        200:
            description: predicted Class
    """

    ## Getting features from swagger UI
    feature_1 = int(request.args.get("feature_1")) # e.g., 1
    feature_2 = int(request.args.get("feature_2")) # e.g., 2
    feature_3 = int(request.args.get("feature_3")) # e.g., 3
    feature_4 = int(request.args.get("feature_4")) # e.g., 4

    test_set = np.array([[feature_1, feature_2, feature_3, feature_4]])

    ## Loading Model
    infile = open('LogisticRegression.pkl','rb')
    model = pickle.load(infile)
    infile.close()

    ## Generating Prediction
    preds = model.predict(test_set)

    return jsonify({"class_name":str(preds)})


@app.route('/predict_default', methods = ['GET'])
def get_predictions_home():

    """
    API that returns the predicted class given four default features as input parameter
    ---

    parameters:
       - name: feature_1
         in: query
         type: number
         required: true
         default: 1

       - name: feature_2
         in: query
         type: number
         required: true
         default: 2

       - name: feature_3
         in: query
         type: number
         required: true
         default: 3

       - name: feature_4
         in: query
         type: number
         required: true
         default: 4

    responses:
        200:
            description: predicted Class
    """

    feature_1 = 1
    feature_2 = 2
    feature_3 = 3
    feature_4 = 4

    test_set = np.array([[feature_1, feature_2, feature_3, feature_4]])

    ## Loading Model
    infile = open('LogisticRegression.pkl','rb')
    model = pickle.load(infile)
    infile.close()

    ## Generating Prediction
    preds = model.predict(test_set)

    return jsonify({"class_name":str(preds)})

if __name__=='__main__':
    app.run(debug = True, host="0.0.0.0")
