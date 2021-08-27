from flask import Flask, render_template, request,jsonify
from numpy import record
import pandas as pd
import json
from prediction import *

app = Flask(__name__)
df = pd.read_csv('Churn_Modelling.csv')

n_post = 50
data_customer = []
for i in range(0,n_post):
    data_customer.append(df.iloc[i,:])

@app.route("/")
def index():
    return render_template('index.html',data = data_customer)

@app.route("/prediction", methods=['POST'])
def get_prediction():
    id = request.form['text']
    customer = df[df['CustomerId']==int(id)]
    RowNumber = str(customer.iloc[0,0])
    CustomerId = str(customer.iloc[0,1])
    Surname = str(customer.iloc[0,2])
    CreditScore = str(customer.iloc[0,3])
    Geography = str(customer.iloc[0,4])
    Gender = str(customer.iloc[0,5])
    Age = str(customer.iloc[0,6])
    Tenure = str(customer.iloc[0,7])
    Balance = str(customer.iloc[0,8])
    NumOfProducts = str(customer.iloc[0,9])
    HasCrCard = str(customer.iloc[0,10])
    IsActiveMember = str(customer.iloc[0,11])
    EstimatedSalary = str(customer.iloc[0,12])

    pred, prob = churn_prediction(customer)
    result = pred+ ' (' + str(round(prob*100,2))+'%)'
    return jsonify({'result':result,
                    'alert':pred,
                    'RowNumber': RowNumber,
                    'CustomerId': CustomerId,
                    'Surname' : Surname,
                    'CreditScore': CreditScore,
                    'Geography': Geography,
                    'Gender': Gender,
                    'Age': Age,
                    'Tenure': Tenure,
                    'Balance': Balance,
                    'NumOfProducts': NumOfProducts,
                    'HasCrCard': HasCrCard,
                    'IsActiveMember': IsActiveMember,
                    'EstimatedSalary': EstimatedSalary
                    })

if __name__ == '__main__':
    app.run(debug=True)