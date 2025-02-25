import pickle
import time
import numpy as np
from flask import Flask, request, render_template

app = Flask(__name__)

# Load the pre-trained model from a pickle file
try:
    with open('house_prediction.pkl', 'rb') as file:
        model = pickle.load(file)
except FileNotFoundError:
    raise FileNotFoundError("Model file 'house_prediction.pkl' not found.")
except Exception as e:
    raise RuntimeError(f"Error loading model: {e}")

@app.route('/index')
def index():
    return render_template('index1.html')


@app.route('/predict', methods=['POST'])
def predict():
    """Handle form submission and return predictions."""
    try:
        # Get inputs from the form
        input1_text = request.form.get('t1')  # First input
        input2_text = request.form.get('t2')  # Second input

        # Validate and convert inputs
        try:
            input1 = float(input1_text)
            input2 = float(input2_text)
        except ValueError:
            return render_template('result1.html', prediction="Invalid input. Please enter numeric values.")

        # Prepare the input for the model
        input_data = [[input1, input2]]

        # Measure prediction time
        start_time = time.time()
        prediction = model.predict(input_data)  # Get the prediction
        end_time = time.time()

        # Ensure prediction is a scalar
        if isinstance(prediction, (list, np.ndarray)):
            prediction = prediction[0]

        # Convert to float for formatting
        prediction = float(prediction)

        # Render the result
        return render_template(
            'result1.html',
            prediction=f"The predicted house price is: {prediction:.2f}",
            processing_time=f"Prediction completed in {end_time - start_time:.2f} seconds."
        )

    except Exception as e:
        # Log the error and render an error message
        print(f"Error occurred: {e}")
        return render_template('result1.html', prediction=f"An error occurred: {str(e)}")


if __name__ == '__main__':
    app.run(debug=True)
