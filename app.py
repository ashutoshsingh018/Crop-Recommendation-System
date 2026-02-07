from flask import Flask,render_template,request
import joblib
import numpy as np

app=Flask(__name__)

model=joblib.load("model.joblib")


def get_prediction(N,P,K,temp,humidity,ph,rainfall):
    features=np.array([[N,P,K,temp,humidity,ph,rainfall]])
    crop=model.predict(features)[0]
    return crop

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict",methods=["POST"])
def predict():
    try:
        N=float(request.form["N"])
        P=float(request.form["P"])
        K=float(request.form["K"])
        temp=float(request.form["temp"])
        humidity=float(request.form["humidity"])
        ph=float(request.form["ph"])
        rainfall=float(request.form["rainfall"])
    
        crop=get_prediction(N,P,K,temp,humidity,ph,rainfall)
        return render_template("result.html",crop=crop,N=N,P=P,K=K,temp=temp,humidity=humidity,ph=ph,rainfall=rainfall)
    except Exception as e:
        return f"Invalid input or error:{str(e)}"
            
   
    
if __name__=="__main__":
    app.run(debug=True,port=8000)
    