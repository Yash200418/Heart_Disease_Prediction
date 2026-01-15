from flask import Flask,jsonify,render_template,request
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import pickle
import os


application = Flask(__name__, static_folder='templates', static_url_path='/static')
app=application 

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
logistic_path = os.path.join(BASE_DIR, 'models', 'logistic.pkl')
scaler_path = os.path.join(BASE_DIR, 'models', 'scaler.pkl')
try:
    with open(logistic_path, 'rb') as f:
        logistic_model = pickle.load(f)
except FileNotFoundError:
    raise FileNotFoundError(f"Model file not found: {logistic_path}")
try:
    with open(scaler_path, 'rb') as f:
        standard_scaler = pickle.load(f)
except FileNotFoundError:
    raise FileNotFoundError(f"Scaler file not found: {scaler_path}")

@app.route("/")
def index():
    return render_template('index.html')


@app.route('/home')
def home():
    return render_template('home.html', result=None)

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():



    
    if request.method =="POST":
        age=int(request.form.get("age"))
        systolic_pressure=int(request.form.get("systolic_pressure"))
        diastolic_pressure=int(request.form.get("diastolic_pressure"))
        physical_activity=int(request.form.get("physical_activity"))
        smoking=int(request.form.get("smoking"))
        diabetes=int(request.form.get("diabetes"))
        alcohol=int(request.form.get("alcohol"))
        total_cholesterol=float(request.form.get("total_cholesterol"))
       
        new_data_scaled = standard_scaler.transform([[age,systolic_pressure,diastolic_pressure,physical_activity,smoking,diabetes,alcohol,total_cholesterol]])
        # Prefer probability for class 1 (presence of heart disease)
        try:
            proba = logistic_model.predict_proba(new_data_scaled)[0][1]
        except Exception:
            # Fallback: some models may not implement predict_proba
            try:
                proba = float(logistic_model.predict(new_data_scaled)[0])
            except Exception:
                proba = 0.0

        proba = float(np.clip(proba, 0.0, 1.0))
        percentage = round(proba * 100.0, 2)
        result_str = f"{percentage}%"

        return render_template('home.html', result=result_str)
    


    else:
        return render_template('home.html', result=None)

if __name__=="__main__":
    app.run(host="0.0.0.0",debug=True)