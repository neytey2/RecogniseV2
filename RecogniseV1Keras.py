# from keras.models import load_model
# import cv2
# import numpy as np

# # Load the face detection classifier
# facedetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# # Load the pre-trained model and class labels
# model = load_model("keras_model.h5")
# class_names = open("labels.txt", "r").readlines()

# # Open the camera
# camera = cv2.VideoCapture(0)

# while True:
#     # Read frame from the camera
#     ret, frame = camera.read()
    
#     if not ret:
#         continue
    
#     # Detect faces in the frame
#     faces = facedetect.detectMultiScale(frame, 1.3, 5)
    
#     for (x, y, w, h) in faces:
#         # Crop the face region
#         face_img = frame[y:y+h, x:x+w]
        
#         # Resize and preprocess the image for the model
#         face_img = cv2.resize(face_img, (224, 224))
#         face_img = (face_img / 255.0)[np.newaxis, ...]

#         # Make a prediction
#         predictions = model.predict(face_img)
#         class_index = np.argmax(predictions)
#         class_label = class_names[class_index]
#         confidence_score = predictions[0][class_index]

#         # Draw rectangle around the face
#         cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
#         # Display the class label above the face
#         cv2.putText(frame, class_label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
#         print("Class:", class_label[2:], end="")
#         print("Confidence Score:", str(np.round(confidence_score * 100))[:-2], "%")
    
#     # Show the frame with annotations
#     cv2.imshow("Webcam Image", frame)

#     # Listen to the keyboard for presses.
#     key = cv2.waitKey(1)
#     if key == 27:  # Check for ESC key
#         break

# # Release the camera and close all windows
# camera.release()
# cv2.destroyAllWindows()


import cv2
import numpy as np
from keras.models import load_model

# Load the pre-trained model and class labels
model = load_model("keras_Model.h5", compile=False)
class_names = [line.strip() for line in open("labels.txt").readlines()]

# Load the face detection classifier
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Open the camera
camera = cv2.VideoCapture(0)

while True:
    # Grab the webcam's image
    ret, frame = camera.read()

    if not ret:
        continue

    # Resize the image
    resized_frame = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)

    image = np.asarray(resized_frame, dtype=np.float32)
    image = (image / 127.5) - 1
    image = image.reshape(1, 224, 224, 3)

    prediction = model.predict(image)
    class_index = np.argmax(prediction)
    class_label = class_names[class_index]
    confidence_score = np.round(prediction[0][class_index] * 100)

    # Print the predicted class and confidence score
    print("Class:", class_label)
    print("Confidence Score:", confidence_score, "%")

    # Detect faces in the original frame
    
    if confidence_score > 70:
        
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)
        
        for (x, y, w, h) in faces:
        # Draw rectangle around the face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Display the predicted class label above the face
            cv2.putText(frame, class_label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

    # Show the frame with annotations
    cv2.imshow("Webcam Image", frame)

    # Listen to the keyboard for presses.
    key = cv2.waitKey(1)
    if key == 27:  # Check for ESC key
        break

# Release the camera and close all windows
camera.release()
cv2.destroyAllWindows()


