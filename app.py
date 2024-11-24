from flask import Flask, render_template, request, redirect, url_for
import joblib
import pandas as pd

app = Flask(__name__)

# Load models and scalers
crop_model = joblib.load('crop_model.pkl')
fertilizer_model = joblib.load('fertilizer_model.pkl')
scaler_crop = joblib.load('scaler_crop.pkl')
scaler_fertilizer = joblib.load('scaler_fertilizer.pkl')

# Load label encoders
crop_label_encoder = joblib.load('label_encoder_crop.pkl')
fertilizer_label_encoder = joblib.load('label_encoder_fertilizer.pkl')

def get_organic_alternatives(fertilizer):
    alternatives = {
        "Urea": ["Compost", "Vermicompost", "Green Manure", "Fish Emulsion", "Seaweed Extract"],
        "DAP": ["Bone Meal", "Rock Phosphate", "Fish Bone Meal", "Organic Compost"],
        "14-35-14": ["Fish Emulsion", "Compost Tea", "Rock Phosphate", "Bone Meal"],
        "28-28": ["Compost", "Green Manure", "Fish Emulsion", "Seaweed Extract"],
        "17-17-17": ["Compost Tea", "Vermicompost", "Rock Phosphate", "Bone Meal"],
        "20-20": ["Compost", "Green Manure", "Fish Emulsion", "Seaweed Extract"],
        "10-26-26": ["Compost", "Bone Meal", "Rock Phosphate", "Fish Bone Meal"]
    }
    return alternatives.get(fertilizer, ["No organic alternatives available."])

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        # Get input data
        N = float(request.form["N"])
        P = float(request.form["P"])
        K = float(request.form["K"])
        temperature = float(request.form["temperature"])
        humidity = float(request.form["humidity"])
        ph = float(request.form["ph"])
        rainfall = float(request.form["rainfall"])
        moisture = float(request.form["moisture"])

        # Prepare inputs for crop recommendation
        crop_input = pd.DataFrame([[N, P, K, temperature, humidity, ph, rainfall]],
                                  columns=["N", "P", "K", "temperature", "humidity", "ph", "rainfall"])
        crop_input_scaled = scaler_crop.transform(crop_input)
        crop_prediction = crop_model.predict(crop_input_scaled)
        crop_name = crop_label_encoder.inverse_transform(crop_prediction)[0]

        # Prepare inputs for fertilizer recommendation
        fertilizer_input = pd.DataFrame([[N, P, K, temperature, humidity, moisture]],
                                        columns=["Nitrogen", "Phosphorous", "Potassium", "Temparature", "Humidity ", "Moisture"])
        fertilizer_input_scaled = scaler_fertilizer.transform(fertilizer_input)
        fertilizer_prediction = fertilizer_model.predict(fertilizer_input_scaled)
        fertilizer_name = fertilizer_label_encoder.inverse_transform(fertilizer_prediction)[0]

        # Get organic alternatives
        alternatives = get_organic_alternatives(fertilizer_name)

        # Render template with prediction results
        return render_template(
            "index.html",
            crop_name=crop_name,
            fertilizer_name=fertilizer_name,
            alternatives=alternatives,
        )

    # Fallback (unlikely to be reached)
    return render_template("index.html")

if __name__ == "__main__":
    app.run(port=3000, debug=True)


