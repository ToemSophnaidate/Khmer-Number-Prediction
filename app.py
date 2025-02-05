import os
from flask import Flask, request, render_template, send_from_directory
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)

# Load the model
model = load_model("khmer_number_model.h5")

# Class labels
class_labels = {
    0: "0, number zero",
    1: "1, number one",
    2: "2, number two",
    3: "3, number three",
    4: "4, number four",
    5: "5, number five",
    6: "6, number six",
    7: "7, number seven",
    8: "8, number eight",
    9: "9, number nine"
}

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Route for serving uploaded images
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/', methods=['GET', 'POST'])
def predict_number():
    if request.method == 'POST':
        # Handle file upload
        if 'file' not in request.files:
            return "No file part"
        file = request.files['file']
        if file.filename == '':
            return "No selected file"
        if file:
            # Save the uploaded file
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)

            # Preprocess the image
            img = image.load_img(file_path, target_size=(64, 64))
            img_array = image.img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # Make prediction
            prediction = model.predict(img_array)
            predicted_class = np.argmax(prediction)  # Get the index of the highest score
            predicted_label = class_labels[predicted_class]

            # Return results
            return render_template(
                'index.html',
                image_path='/uploads/' + file.filename,  # Updated to use custom route for image
                predicted_label=predicted_label
            )
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
