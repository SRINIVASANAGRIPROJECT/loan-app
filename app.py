from flask import Flask, request, jsonify
import pickle
import numpy as np
import os
app = Flask(__name__)

model = pickle.load(open('loan_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    
    # Convert inputs (same as your training)
    gender = 1 if data['Gender'] == "Male" else 0
    married = 0 if data['Married'] == "Yes" else 1
    dependents = int(data['Dependents'])
    education = 0 if data['Education'] == "Graduate" else 1
    self_emp = 0 if data['Self_Employed'] == "Yes" else 1

    applicant_income = float(data['ApplicantIncome'])
    coapplicant_income = float(data['CoapplicantIncome'])
    loan_amount = float(data['LoanAmount'])
    credit_history = float(data['Credit_History'])

    if data['Property_Area'] == "Semiurban":
        property_area = 0
    elif data['Property_Area'] == "Urban":
        property_area = 1
    else:
        property_area = 2

    features = np.array([[gender, married, dependents, education,
                          self_emp, applicant_income,
                          coapplicant_income, loan_amount,
                          credit_history, property_area]])

    # Apply scaling
    features[:, [5,6,7]] = scaler.transform(features[:, [5,6,7]])

    pred = model.predict(features)

    return jsonify({
        "result": "Approved" if pred[0] == 0 else "Rejected"
    })



port = int(os.environ.get("PORT", 10000))
app.run(host="0.0.0.0", port=port)
