from flask import Flask, render_template, request,  send_from_directory
import cv2
import numpy as np
import os
from keras.models import model_from_json

# Initialize Flask app
app = Flask(__name__)

# Create a folder to temporarily save uploaded images
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    raise FileNotFoundError(f"The upload folder '{UPLOAD_FOLDER}' does not exist. Please create it manually.")
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


emotion_verdict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}


with open('../detection_model.json', 'r') as json_file:
    loaded_model_json = json_file.read()
detection_model = model_from_json(loaded_model_json)
detection_model.load_weights("../detection_model.h5")
print("Model loaded from disk")


face_detector = cv2.CascadeClassifier('../haarcascades/haarcascade_frontalface_default.xml')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_emotion():
    if 'image' not in request.files:
        return render_template('index.html', error="No image uploaded. Please upload an image.")

    file = request.files['image']
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)

    frame = cv2.imread(file_path)
    if frame is None:
        return render_template('index.html', error="Invalid image. Please upload a valid image.")

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    num_faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

    if len(num_faces) == 0:
        return render_template('index.html', error="No faces detected in the image.", image_path=file.filename)

    results = []
    for (x, y, w, h) in num_faces:
        roi_gray_frame = gray_frame[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)

        emotion_prediction = detection_model.predict(cropped_img)
        maxindex = int(np.argmax(emotion_prediction))
        emotion_label = emotion_verdict[maxindex]
        
        cv2.rectangle(frame, (x, y - 50), (x + w, y + h + 10), (0, 255, 0), 2)
        cv2.putText(frame, f"{emotion_label}", (x + 5, y - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2, cv2.LINE_AA)

        results.append({"emotion": emotion_label})

    # Saving processed image
    processed_image_path = os.path.join(app.config['UPLOAD_FOLDER'], f"processed_{file.filename}")
    cv2.imwrite(processed_image_path, frame)

    # Rendering the results on the webpage
    return render_template('results.html', results=results, image_path=f"processed_{file.filename}")

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
