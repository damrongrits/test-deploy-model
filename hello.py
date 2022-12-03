
import pickle
import numpy as np
import pandas as pd

from flask import Flask, request

app = Flask(__name__)

model = pickle.load(open('dealsDecisionTree.pk', 'rb'))
deals = pd.read_csv("/Users/anos/Documents/Works/Training/dataset/Deals.csv")
feature_cols = ['Age', 'Gender', 'Payment Method']
X_train = deals[feature_cols] # Features
X_train = pd.get_dummies(X_train)

@app.route("/hello_world")
def hello_world():
    return "<p>Hello, World!</p>"

@app.route('/prediction', methods = ['POST', 'GET'])
def prediction():
    if request.method == 'GET':
        return 'Please Resend with POST Method with Attributes'
    
    elif request.method == 'POST':
        X = request.get_json()
        
        x1 = X['x1']
        x2 = X['x2']
        x3 = X['x3']

        X_call = pd.DataFrame({'Age':[x1],
                       'Gender': [x2],
                       'Payment Method': [x3] })
        X_call = pd.get_dummies(X_call)
        X_call = X_call.reindex(columns = X_train.columns, fill_value=0)

        #XTest = np.array([[x1, x2, x3]])

        predicted = model.predict(X_call)[0]
        #predicted = 1 / (1 + np.exp(-predicted))
        return  predicted

if __name__ == '__main__':
    app.run(debug = True)