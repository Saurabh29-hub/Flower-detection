from flask import Flask, request, render_template
import numpy as np
import pickle

app = Flask(__name__)

# Load the trained model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template('form.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get features from form input
        features = [float(x) for x in request.form.values()]
        prediction = model.predict(np.array([features]))[0]

        # Map prediction to name and image
        class_map = {
            0: ('Iris Setosa', 'setosa.jpg'),
            1: ('Iris Versicolor', 'versicolor.jpg'),
            2: ('Iris Virginica', 'virginica.jpg')
        }

        flower_name, image_file = class_map[prediction]

        return render_template('form.html',
                               prediction=prediction,
                               flower_name=flower_name,
                               image_file=image_file)

    except Exception as e:
        return render_template('form.html',
                               prediction="Error",
                               flower_name="",
                               image_file="")
        
if __name__ == '__main__':
    app.run(debug=True)
