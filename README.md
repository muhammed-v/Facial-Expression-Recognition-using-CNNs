
# Facial Expression Recognition using CNNs


### Packages need to be installed
- python -m venv .venv (then, select the appropriate interpreter)
- pip install numpy
- pip install opencv-python / pip install opencv-python --no-cache-dir
- pip install keras
- pip3 install --upgrade tensorflow
- pip install pillow
- pip install scipy
- pip install scikit-learn matplotlib seaborn

### Download FER2013 dataset
- Download from below link and paste in dataset folder under the project directory
- https://www.kaggle.com/msambare/fer2013

### Other instructions
- `sample-images` folder contains the sample images for emotion detection
- `screenshots` and `model-training-logs.txt`folder contain the required screenshots of training logs, validation results, and final performance metrics.
- `emotion-model-training.py` is used for model training. After training, detection_model.h5 and detection_model.json are created.
- `emotion-detector.py` is used to run the emotion detection software.
- `web-app-version` folder contains the web app for uploading images and predicting the emotions within them. 
- web-app-version/static folder contains the stylesheet for the web app. 
- web-app-version/templates folder contains the html files.
- web-app-version/uploads folder stores the uploaded images and the processed images formed after emotion detection.
- `python app.py` to run the web application
  






## Problem Statement

The task is to build a model that classifies facial expressions (such as happy, sad, neutral, etc.) from images.





## Approach


### 1. Dataset
The `FER 2013` dataset was used for training and evaluation. The dataset contains 48x48 grayscale images of human faces labeled with seven different emotion categories:  
- Angry
- Disgusted
- Fearful
- Happy
- Neutral
- Sad
- Surprised


### 2. Data Preprocessing
- Images are resized to 48x48 pixels for consistency.
- Grayscale images are used as the color channels are not essential for facial expression recognition.
- The data is loaded using `ImageDataGenerator` from Keras, which automatically scales pixel values to a range between 0 and 1 using `rescale=1/255`.


### 3. CNN Model Architecture
A Convolutional Neural Network (CNN) was built to classify the facial expressions. The architecture consists of:
- Convolutional Layers: To capture features from the images.
- Max-Pooling Layers: To reduce spatial dimensions.
- Dropout Layers: To prevent overfitting.
- Dense Layers: For classification with a final softmax layer to output probabilities for the seven emotion categories.


### 4. Model Compilation & Training
- The model is compiled using categorical cross-entropy loss, suitable for multi-class classification, and the Adam optimizer with a small learning rate.
- The model was trained for 60 epochs using a batch size of 64 images, with separate training and validation datasets.


### 5. Model Evaluation
After training, the model was evaluated using:
- Accuracy: The percentage of correct predictions.
- F1-Score: The harmonic mean of precision and recall, useful for imbalanced datasets.
- Confusion Matrix: To show the distribution of predicted and true labels.
- Classification Report: To provide detailed metrics like precision, recall, and F1-score for each class.


### 6. Productization (Bonus)
The trained model was deployed via a Flask web application that allows users to upload an image for facial expression prediction. The app processes the image, detects faces using a Haar Cascade Classifier, and classifies the detected facial expressions. The results are displayed on the web interface.





## Results


### Performance Metrics:
- Validation Accuracy: The model achieved an `accuracy of 0.6308163833937029` on the validation dataset.
- F1-Score:
	Angry - 0.53, Disgusted - 0.68, Fearful - 0.49, Happy - 0.82, Neutral - 0.58, Sad - 0.50, Surprised - 0.78

  

### Visualizations:
- `Confusion Matrix`: Shows the accuracy of predictions across different classes. 
  
	REFER SCREENSHOTS
  
- `Correctly Classified and Misclassified Images`: Samples from the validation dataset where the model correctly or incorrectly predicted emotions. 

	REFER SCREENSHOTS


### Observations:
- The more the Epoch, the better the model becomes at emotion detection
- The model performs well on emotions with clear facial expressions (e.g., happy, sad) but struggles with more subtle expressions (e.g., neutral).
- The confusion matrix shows some misclassifications between emotions with similar facial features, like happy vs. surprised.


## Challenges

- Finding the right Epoch for emotion detection. 
    Epoch very large-> takes more time to train and heavy load on laptop
    Epoch very small-> the model gives less accurate results.
- Some emotions, such as disgusted and fearful, were underrepresented in the dataset, which affected the model’s ability to predict these emotions accurately.
- The Haar Cascade Classifier sometimes failed to detect faces in certain poses or angles.
  
