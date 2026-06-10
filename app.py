from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# Load model
model = joblib.load("penguin_rf_model.pkl")
encoder = joblib.load("species_encoder.pkl")


@app.route("/")
def home():
    return render_template("random.html")


@app.route("/predict", methods=["POST"])
def predict():

    bill_length = float(request.form["bill_length"])
    bill_depth = float(request.form["bill_depth"])
    flipper_length = float(request.form["flipper_length"])
    body_mass = float(request.form["body_mass"])
    sex = int(request.form["sex"])
    island = int(request.form["island"])

    features = np.array([[bill_length, bill_depth, flipper_length,
                          body_mass, sex, island]])

    prediction = model.predict(features)

    species = encoder.inverse_transform(prediction)

    return render_template(
        "random.html",
        prediction_text="Predicted Penguin: " + species[0]
    )


if __name__ == "__main__":
    app.run(debug=True)