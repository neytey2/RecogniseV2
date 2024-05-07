from hmac import new
import os
import time
import uuid
import cv2

IMAGES_PATH = os.path.join('images')
counter = 0
number_images = 100

#
cap = cv2.VideoCapture(0);

for imgnum in range (number_images):
    print('Collecting image {}'.format(imgnum))
    ret, frame = cap.read()
    imgname = os.path.join(IMAGES_PATH,f'{counter}.jpg')
    cv2.imwrite(imgname, frame)
    cv2.imshow('frame', frame)
    time.sleep(0.1)
    counter = counter + 1
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()