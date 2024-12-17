import cv2
import numpy as np
from keras.models import model_from_json

# Emotion label mapping
emotion_verdict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# Loading model architecture and weights
json_file = open('./detection_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
detection_model = model_from_json(loaded_model_json)
detection_model.load_weights("./detection_model.h5")
print("Loaded model from disk")

# Loading Haar Cascade for face detection
face_detector = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')

# Loading image (replace with the path to your image)
image_path = "./sample-images/sample1.jpg"
frame = cv2.imread(image_path)

if frame is None:
    print("Error: Unable to load image. Please check the path.")
else:
    # Resizing frame for consistent display
    frame = cv2.resize(frame, (1280, 720))
    
    # Converting image to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detecting faces in the image
    num_faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

    # Processing each detected face
    for (x, y, w, h) in num_faces:
        # Draw the bounding box
        cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (0, 255, 0), 4)
        
        # Extracting face ROI and preprocessing it for prediction
        roi_gray_frame = gray_frame[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)

        # Predicting emotion
        emotion_prediction = detection_model.predict(cropped_img)
        maxindex = int(np.argmax(emotion_prediction))

        # Adding emotion label to the image
        cv2.putText(frame, emotion_verdict[maxindex], (x+5, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    # Displaying the result until a key is pressed (for exit)
    cv2.imshow('Emotion Detection', frame)
    print("Press any key to close the window...")
    cv2.waitKey(0)  # Wait indefinitely for a key press
    cv2.destroyAllWindows()
