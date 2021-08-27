from app import index
import joblib
import pandas as pd
import numpy as np

model = joblib.load('xgb_model.pkl')
encoder = joblib.load('encoder.pkl')
features = joblib.load('features_selected.pkl')
sc = joblib.load('standardscaler.pkl')

def churn_prediction(series):
    column_numeric = ['CreditScore','Age','Balance','EstimatedSalary']
    column_category = ['Geography','Gender','Tenure','NumOfProducts','HasCrCard','IsActiveMember']
    thresholdOpt = 0.3366

    type_changed = series.drop(columns=['RowNumber','CustomerId','Surname','Exited'])
    #type_changed = idx_droped.astype({"Geography":'category',"Gender":'category',"Tenure":'category',"NumOfProducts":'category',"HasCrCard":'category',"IsActiveMember":'category',"Exited":'category'})
    
    data = type_changed.iloc[0,:]
    #Data scaling
    numeric = data[column_numeric]
    numeric_arr = np.array(numeric).reshape(1,-1)
    std = sc.transform(numeric_arr)
    for i,col in enumerate(column_numeric):
        numeric[col] = std[0][i]

    #Data encoding
    category = data[column_category]
    category_arr = np.array(category).reshape(1,-1)
    encoded = encoder.transform(category_arr)
    fe = numeric
    for i,col in enumerate(encoder.get_feature_names(column_category)):
        fe[col] = encoded[0][i]

    #feature selection
    features_selected = fe[features]

    pred = model.predict_proba(np.array(features_selected).reshape(1,-1))

    if pred[0,1] >= thresholdOpt:
        result = "Churn"
        prob = pred[0,1]
    else:
        result = "Stay"
        prob = pred[0,0]
    
    return result, prob