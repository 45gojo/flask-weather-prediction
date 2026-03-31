from flask import Flask, render_template, request
import joblib
import numpy as np

application = Flask(__name__)

model = joblib.load(r"C:\Users\anshk\Desktop\ML\projects\Wether_prediction\xg_model.joblib")

@application.route("/", methods=["GET", "POST"])
def home():
    prediction_text = None

    if request.method == "POST":
        try:
            MaxT = float(request.form['MaxT'])
            MinT = float(request.form['MinT'])
            RH1 = float(request.form['RH1'])
            RH2 = float(request.form['RH2'])
            Wind = float(request.form['Wind'])
            SSH = float(request.form['SSH'])
            Evap = float(request.form['Evap'])
            Radiation = float(request.form['Radiation'])
            Month = int(request.form['Month'])
            Year = int(request.form['Year'])
            Day = int(request.form['Day'])
            Rain_lag1 = float(request.form['Rain_lag1'])
            Rain_lag2 = float(request.form['Rain_lag2'])
            Rain_3days = np.log1p(float(request.form['Rain_3days']))  # log transform
            Rain_7days = np.log1p(float(request.form['Rain_7days']))  # log transform

            
            features = np.array([[MaxT, MinT, RH1, RH2, Wind, SSH, Evap, Radiation,
                                  Month, Year, Day, Rain_lag1, Rain_lag2, Rain_3days, Rain_7days]])

           
            prediction = model.predict(features)
            if prediction[0] == 1:
                prediction_text = "Yes, it will rain tomorrow! ☔"
            else:
                prediction_text = "No rain tomorrow! 🌞"

        except Exception as e:
            prediction_text = f"Error: {e}"

    return render_template("index.html", prediction_text=prediction_text)

if __name__ == "__main__":
    application.run(debug=True, host="127.0.0.1", port=8000)