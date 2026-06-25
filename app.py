import pickle
import numpy as np
from flask import Flask,request, jsonify
# Initialize the Flask application instance
app = Flask(__name__)   
# Load the pre-trained machine learning model into memory
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def home():
    return "Hello from Flask! 🚀"

@app.route('/health')
def health():
    return "OK"

@app.route('/test')
def test():
    return "It looks cool!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract the JSON payload array sent by the client
        data = request.get_json()
        
        # Convert inputs into a 2D NumPy array structure required by scikit-learn
        prediction_inputs = np.array(data['features']).reshape(1, -1)
        
        # Execute model inference
        prediction = model.predict(prediction_inputs)
        
        # Return the resulting numerical target mapping as a JSON response
        return jsonify({'prediction': int(prediction[0])})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    # Start the Flask local development server
    app.run(host='0.0.0.0', port=5000, debug=True)

