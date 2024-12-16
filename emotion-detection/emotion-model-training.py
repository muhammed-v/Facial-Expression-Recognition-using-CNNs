# Importing Required Libraries
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from keras.optimizers.legacy import Adam
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Initialization of image data generator with rescaling
train_data_gen = ImageDataGenerator(rescale=1.0 / 255)
validation_data_gen = ImageDataGenerator(rescale=1.0 / 255)

# Defining the dataset paths
train_path = "dataset/train"
test_path = "dataset/test"

#  Preprocessing of train folder images
train_generator = train_data_gen.flow_from_directory(
    train_path,
    target_size=(48, 48),
    batch_size=64,
    color_mode="grayscale",
    class_mode="categorical",
)

# Preprocessing of test folder images
validation_generator = validation_data_gen.flow_from_directory(
    test_path,
    target_size=(48, 48),
    batch_size=64,
    color_mode="grayscale",
    class_mode="categorical",
)

# Defining emotion classes
emotion_verdict = {
    0: "Angry",
    1: "Disgusted",
    2: "Fearful",
    3: "Happy",
    4: "Neutral",
    5: "Sad",
    6: "Surprised",
}

# Creating the CNN model
emotion_model = Sequential()

emotion_model.add(Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=(48, 48, 1)))
emotion_model.add(Conv2D(64, kernel_size=(3, 3), activation="relu"))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))

emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation="relu"))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation="relu"))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))

emotion_model.add(Flatten())
emotion_model.add(Dense(1024, activation="relu"))
emotion_model.add(Dropout(0.5))
emotion_model.add(Dense(7, activation="softmax"))

cv2.ocl.setUseOpenCL(False)


emotion_model.compile(
    loss="categorical_crossentropy",
    optimizer=Adam(learning_rate=0.0001, decay=1e-6),
    metrics=["accuracy"],
)

# Training the neural network/model
emotion_model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=60,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
)

# Saving the model
model_json = emotion_model.to_json()
with open("detection_model.json", "w") as json_file:
    json_file.write(model_json)
emotion_model.save_weights("detection_model.h5")

print("Model training completed and saved!")

# Evaluation of the model
print("\nEvaluating the model on the validation set...")
val_labels = []
val_predictions = []
for i in range(len(validation_generator)):
    x_val, y_val = validation_generator[i]
    predictions = emotion_model.predict(x_val)
    val_predictions.extend(np.argmax(predictions, axis=1))
    val_labels.extend(np.argmax(y_val, axis=1))

# Computing Accuracy and Classification Report
print("Validation Accuracy:", accuracy_score(val_labels, val_predictions))
print("\nClassification Report:")
print(classification_report(val_labels, val_predictions, target_names=emotion_verdict.values()))

# Confusion Matrix
conf_matrix = confusion_matrix(val_labels, val_predictions)
plt.figure(figsize=(10, 8))
sns.heatmap(
    conf_matrix,
    annot=True,
    fmt="d",
    xticklabels=emotion_verdict.values(),
    yticklabels=emotion_verdict.values(),
    cmap="Blues",
)
plt.title("Confusion Matrix")
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.show()

# Correctly classified and misclassified images.
def visualize_images(indices, generator, title, rows=3, cols=5):
    plt.figure(figsize=(15, 10))
    plt.suptitle(title, fontsize=16)
    for i, idx in enumerate(indices[: rows * cols]):
        x_val, y_val = generator[idx // generator.batch_size]
        image = x_val[idx % generator.batch_size]
        true_label = np.argmax(y_val[idx % generator.batch_size])
        plt.subplot(rows, cols, i + 1)
        plt.imshow(image.squeeze(), cmap="gray")
        plt.title(f"True: {emotion_verdict[true_label]}\nPred: {emotion_verdict[val_predictions[idx]]}")
        plt.axis("off")
    plt.tight_layout()
    plt.show()


correct_indices = [i for i, (y, pred) in enumerate(zip(val_labels, val_predictions)) if y == pred]
incorrect_indices = [i for i, (y, pred) in enumerate(zip(val_labels, val_predictions)) if y != pred]

# Correctly Classified Images
visualize_images(correct_indices, validation_generator, "Correctly Classified Images")

# Misclassified Images
visualize_images(incorrect_indices, validation_generator, "Misclassified Images")
