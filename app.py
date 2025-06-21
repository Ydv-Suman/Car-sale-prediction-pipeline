from flask import Flask, request, render_template
import pandas as pd
import numpy as np
import pickle

app = Flask(__name__)

# Load the model
try:
    with open("car-sales-prediction-pipeline.pkl", "rb") as f:
        model = pickle.load(f)
    print("✅ Model loaded successfully")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit()

@app.route('/')
def home():
    # These should match your training data categories
    makes = ['Toyota', 'Honda', 'Nissan', 'BMW', 'Missing']
    colors = ['Black', 'White', 'Red', 'Blue', 'Missing']
    return render_template('index.html', makes=makes, colors=colors)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get and validate form data
        make = request.form.get('make', 'Missing')
        color = request.form.get('color', 'Missing')
        
        # Ensure odometer is float (matches training data)
        try:
            odometer = float(request.form.get('odometer', 0))
        except:
            odometer = 0.0  # Default value if conversion fails
            
        # Ensure doors is int (matches training data)
        try:
            doors = int(request.form.get('doors', 4))
        except:
            doors = 4  # Default value if conversion fails

        # Create input data with correct dtypes
        input_data = pd.DataFrame({
            'Make': [make],
            'Color': [color],
            'Odometer(Miles)': [odometer],  # Will be float64
            'Doors': [doors]  # Will be int64
        })

        # Convert Doors to float to match training pipeline
        input_data['Doors'] = input_data['Doors'].astype(float)

        # Make prediction
        prediction = model.predict(input_data)[0]

        # Format results
        result = {
            'make': make,
            'color': color,
            'odometer': f"{odometer:,.1f} miles",
            'doors': doors,
            'price': f"${prediction:,.2f}"
        }

        return render_template('result.html', **result)

    except Exception as e:
        return render_template('error.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True)