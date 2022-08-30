import pickle;
import flask ;
from flask import Flask,request,app,jsonify,url_for,render_template ;
import numpy as np ;
import pandas as pd;
import json

app=Flask(__name__)
##Uploaded model
model =pickle.load(open('LR.pkl','rb'))
######
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api',methods =['POST'])
def predict_api():
    data=request.json['data']
    print(data)
    # Accesing values of json 
    data_values=(data.values())
    #converint into list
    data_values =list(data_values)
    data_values=np.array(data_values)
    data_values=data_values.reshape(1,-1)
    # we can do scaling if required
    output=model.predict(data_values.tolist()).tolist()
    #output=model.predict(data_values)
    print(output)
    return jsonify(output[0])

@app.route('/predict',methods=["POST"])
def predict(): 
    data=[float(x) for x in request.form.values()]
    final_input =np.array(data).reshape(1,-1)
    output=model.predict(final_input)[0]
    return render_template("home.html",prediction_text='Heart may fail if result is 1 ,Heart is Healthy if result is 0 & Result of Heart state ={}'.format(output))

if __name__=="__main__":
    app.run(debug=True)



