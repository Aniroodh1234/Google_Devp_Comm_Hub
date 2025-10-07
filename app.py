from flask import Flask, render_template, request
import pickle
import numpy as np
import os

# Initialize Flask app
app = Flask(__name__)

# Load trained model
MODEL_PATH = "logistic_regression_model/log_reg_model.pkl"
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# Route for homepage
@app.route('/')
def home():
    return render_template('Logistic_regression_GDG.html')

# Route to handle form submission
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        features = [float(x) for x in request.form.values()]
        final_features = np.array(features).reshape(1, -1)

        # Predict using the model
        prediction = model.predict(final_features)[0]
        prob = model.predict_proba(final_features).tolist()[0]

        # Create readable output
        output_text = f"Predicted Class: {prediction} (Confidence: {max(prob)*100:.2f}%)"

        return render_template('Logistic_regression_GDG.html', prediction_text=output_text)
    
    

    except Exception as e:
        return render_template('Logistic_regression_GDG.html', prediction_text=f"Error: {e}")

if __name__ == "__main__":
    app.run(debug=True)
