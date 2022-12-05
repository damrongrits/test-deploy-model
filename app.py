
import pickle
import numpy as np
import pandas as pd

from flask import Flask, request, render_template

app = Flask(__name__)

model = pickle.load(open('dealsDecisionTree.pk', 'rb'))
deals = pd.read_csv("./Deals.csv")
feature_cols = ['Age', 'Gender', 'Payment Method']
X_train = deals[feature_cols] # Features
X_train = pd.get_dummies(X_train)

@app.route('/')
def main():
    return render_template('index.html')

@app.route('/predict', methods = ['POST'])
def getPredict():
    x1 = request.form['x1']
    x2 = request.form['x2']
    x3 = request.form['x3']

    X_call = pd.DataFrame({'Age':[x1], 'Gender': [x2], 'Payment Method': [x3] })
    X_call = pd.get_dummies(X_call)
    X_call = X_call.reindex(columns = X_train.columns, fill_value=0)

    predicted = model.predict(X_call)[0]

    result = {
        "Age":x1,
        "Gender":x2,
        "Payment Method":x3,
        "Predicted (Future Customer)":predicted
    }

    return render_template('index.html', prediction_text = result)
    #return render_template('index.html', prediction_text = f'Predicted (Future Customer): {predicted}')

if __name__ == '__main__':
    app.run(debug = True)