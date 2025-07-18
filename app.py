from flask import Flask,request,render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipelines.predict_pipeline import CustomData,PredictPipeline


app = Flask(__name__)

## Route for home page
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET','POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        data=CustomData(
            LIMIT_BAL = float(request.form['LIMIT_BAL']),
            SEX = int(request.form['SEX']),
            EDUCATION = int(request.form['EDUCATION']),
            MARRIAGE = int(request.form['MARRIAGE']),
            AGE = int(request.form['AGE']),
            PAY_0 = int(request.form['PAY_0']),
            PAY_2 = int(request.form['PAY_2']),
            PAY_3 = int(request.form['PAY_3']),
            PAY_4 = int(request.form['PAY_4']),
            PAY_5 = int(request.form['PAY_5']),
            PAY_6 = int(request.form['PAY_6']),
            BILL_AMT1 = float(request.form['BILL_AMT1']),
            BILL_AMT2 = float(request.form['BILL_AMT2']),
            BILL_AMT3 = float(request.form['BILL_AMT3']),
            BILL_AMT4 = float(request.form['BILL_AMT4']),
            BILL_AMT5 = float(request.form['BILL_AMT5']),
            BILL_AMT6 = float(request.form['BILL_AMT6']),
            PAY_AMT1 = float(request.form['PAY_AMT1']),
            PAY_AMT2 = float(request.form['PAY_AMT2']),
            PAY_AMT3 = float(request.form['PAY_AMT3']),
            PAY_AMT4 = float(request.form['PAY_AMT4']),
            PAY_AMT5 = float(request.form['PAY_AMT5']),
            PAY_AMT6 = float(request.form['PAY_AMT6']),
        )
        pred_df = data.get_data_as_data_frame()
        print(pred_df)

        predict_pipeleine=PredictPipeline()
        prediction= predict_pipeleine.predict(pred_df)
        result = "Default" if prediction[0] == 1 else "Not Default"
        return render_template('home.html',result=result)
        
if __name__ == "__main__":
    app.run(debug=True)    
