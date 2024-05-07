from keras.models import load_model  # Importing Keras from Tensorflow Library
import cv2  # Importing opencv-python
import numpy as np  # Importing Numpy as np

np.set_printoptions(suppress=True)

# Load Keras model
model = load_model("newKeras/keras_Model.h5", compile=False)

# Load text labels
class_names = open("newKeras/labels.txt", "r").readlines()

# Load Computer Camera
camera = cv2.VideoCapture(0)

# Set the width and height of the displayed image
display_width = 640
display_height = 480

while True:
    # Get Image from Webcam
    ret, image = camera.read()

    # Resizing the Webcams image to desired size for display
    image_display = cv2.resize(image, (display_width, display_height), interpolation=cv2.INTER_AREA)

    # Resizing the Webcams image to match model input shape
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)

    # Showing the image from Webcam on screen
    cv2.imshow("Webcam Image", image_display)

    # Make the image a numpy array and reshape it to the models input shape.
    image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)

    # Normalize numpy image array
    image = (image / 127.5) - 1

    # Model Predicting the Image
    prediction = model.predict(image)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    # Show predicted class and probability score on cmd
    print("Class:", class_name[2:], end="")
    print("Confidence Score:", str(np.round(confidence_score * 100))[:-2], "%")

    # Listen to the keyboard for presses.
    keyboard_input = cv2.waitKey(1)

    # Break on ESC pressed
    if keyboard_input == 27:
        break

camera.release()
cv2.destroyAllWindows()
