# app.py
from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load model
model = joblib.load("model.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Collect form data
        Age = int(request.form['Age'])
        Gender = 1 if request.form['Gender'].lower() == 'male' else 0
        Total_Bilirubin = float(request.form['Total_Bilirubin'])
        Direct_Bilirubin = float(request.form['Direct_Bilirubin'])
        Alkaline_Phosphotase = float(request.form['Alkaline_Phosphotase'])
        Alamine_Aminotransferase = float(request.form['Alamine_Aminotransferase'])
        Aspartate_Aminotransferase = float(request.form['Aspartate_Aminotransferase'])
        Total_Protiens = float(request.form['Total_Protiens'])
        Albumin = float(request.form['Albumin'])
        Albumin_and_Globulin_Ratio = float(request.form['Albumin_and_Globulin_Ratio'])

        # Derived features
        Bilirubin_Ratio = Direct_Bilirubin / (Total_Bilirubin + 1e-5)
        Enzyme_Sum = Alamine_Aminotransferase + Aspartate_Aminotransferase
        Protein_Gap = Total_Protiens - Albumin

        # Log transformation
        TB_log = np.log1p(Total_Bilirubin)
        ALP_log = np.log1p(Alkaline_Phosphotase)
        ALT_log = np.log1p(Alamine_Aminotransferase)
        AST_log = np.log1p(Aspartate_Aminotransferase)

        input_data = pd.DataFrame([{
            'Age': Age,
            'Gender': Gender,
            'Total_Bilirubin': TB_log,
            'Direct_Bilirubin': Direct_Bilirubin,
            'Alkaline_Phosphotase': ALP_log,
            'Alamine_Aminotransferase': ALT_log,
            'Aspartate_Aminotransferase': AST_log,
            'Total_Protiens': Total_Protiens,
            'Albumin': Albumin,
            'Albumin_and_Globulin_Ratio': Albumin_and_Globulin_Ratio,
            'Bilirubin_Ratio': Bilirubin_Ratio,
            'Enzyme_Sum': Enzyme_Sum,
            'Protein_Gap': Protein_Gap
        }])

        pred = model.predict(input_data)[0]
        prob = model.predict_proba(input_data)[0]
        result = f"{' NO LIVER DISEASE' if pred == 1 else ' LIVER DISEASE PRESENT'} ({prob[pred]*100:.2f}% confidence)"

        return render_template("index.html", result=result)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
