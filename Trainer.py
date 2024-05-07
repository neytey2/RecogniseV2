from re import T
from hmac import new
import os
import time
import uuid
import albumentations as alb
import enum
import tensorflow as tf
import cv2
import json
import numpy as np
from matplotlib import pyplot as plt

# Load the cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# To capture video from webcam.
# cap = cv2.VideoCapture(0)
# # To use a video file as input
# # cap = cv2.VideoCapture('filename.mp4')

# while True:
#     # Read the frame
#     _, img = cap.read()

#     # Convert to grayscale
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#     # Detect the faces
#     faces = face_cascade.detectMultiScale(gray, 1.1, 4)

#     # Draw the rectangle around each face
#     for (x, y, w, h) in faces:
#         cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

#     # Display
#     cv2.imshow('img', img)

#     # Stop if escape key is pressed
#     k = cv2.waitKey(30) & 0xff
#     if k==27:
#         break

# # Release the VideoCapture object
# cap.release()

#--------------------------------------------------------------------------------------------------------------------------------------------


# # Review Database and Build Image Loading Function


#Avoid OOM errors by setting GPU Memory Consumption Growth
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
    
tf.config.list_physical_devices('GPU')


# Load Image into TF Data Pipeline
images = tf.data.Dataset.list_files('images\\*.jpg', shuffle=False)

def load_image(x):
    byte_img = tf.io.read_file(x)
    img = tf.io.decode_jpeg(byte_img)
    return img

images = images.map(load_image)

images.as_numpy_iterator().next()

type(images)


images_generator = images.batch(4).as_numpy_iterator()
plot_images = images_generator.next()

fig, ax = plt.subplots(ncols=4, figsize=(20,20))
for idx, image in enumerate(plot_images):
    ax[idx].imshow(image)
plt.show() 

#--------------------------------------------------------------------------------------------------------------------------------------------

#3. Partition Unaugmented Data
for folder in ['train', 'test', 'val']:
    for file in os.listdir(os.path.join(folder,'images')):
        
        filename = file.split('.')[0]+'.json'
        existing_filepath = os.path.join('labels',filename)
        if os.path.exists(existing_filepath):
            new_filepath = os.path.join(folder,'labels',filename)
            os.replace(existing_filepath,new_filepath)

#---------------------------------------------------------------------------------------------------------------------------------------------

#4. Apply Image Augmentation on Images and Labels using Albumentations

augmentor = alb.Compose([alb.RandomCrop(width=450, height=450),
                         alb.HorizontalFlip(p=0.5),
                         alb.RandomBrightnessContrast(p=0.2),
                         alb.RandomGamma(p=0.2),
                         alb.RGBShift(p=0.2),
                         alb.VerticalFlip(p=0.5)],
                        bbox_params = alb.BboxParams(format='albumentations', label_fields=['class_labels']))

#4.2 Load a Test Image and Annotations with OpenCV and JSON

img = cv2.imread(os.path.join('train', 'images', '0.jpg'))

with open(os.path.join('train', 'labels', '0.json'), 'r') as f:
    label = json.load(f)
    
label['shapes'][0]['points']
    
#4.3 Extract Coordinates and Rescale to Match Image Resolution
coords = [0,0,0,0]
coords[0] = label['shapes'][0]['points'][0][0]
coords[1] = label['shapes'][0]['points'][0][1]
coords[2] = label['shapes'][0]['points'][1][0]
coords[3] = label['shapes'][0]['points'][1][1]

coords = list(np.divide(coords, [640,480,640,480]))

#4.4 Apply Augmentations and View Results

augmented = augmentor(image=img, bboxes=[coords], class_labels=['Mensch'])

cv2.rectangle(augmented['image'],
              tuple(np.multiply(augmented['bboxes'][0][:2], [450,450]).astype(int)),
              tuple(np.multiply(augmented['bboxes'][0][2:], [450,450]).astype(int)),
              (255,0,0), 2)

plt.imshow(augmented['image'])

#---------------------------------------------------------------------------------------------------------------------------------------------

#5. Build and Run Augmentation Pipeline

#5.1 Run Augmentation Pipeline
for partition in ['train', 'test', 'val']:
    for image in os.listdir(os.path.join(partition, 'images')):
        img = cv2.imread(os.path.join(partition, 'images', image))
        
        coords = [0,0,0.00001,0.00001]
        label_path = os.path.join(partition, 'labels', f'{image.split(".")[0]}.json')
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                label = json.load(f)
                
            coords[0] = label['shapes'][0]['points'][0][0]
            coords[1] = label['shapes'][0]['points'][0][1]
            coords[2] = label['shapes'][0]['points'][1][0]
            coords[3] = label['shapes'][0]['points'][1][1]
            coords = list(np.divide(coords, [640,480,640,480]))
        
        try:
            for x in range(1):
               augmented = augmentor(image=img, bboxes=[coords], class_labels=['Mensch'])
               cv2.imwrite(os.path.join('aug_data', partition, 'images', f'{image.split(".")[0]}.{x}.jpg'), augmented['image'])
               
               annotation = {}
               annotation['image'] = image
               
               if os.path.exists(label_path):
                   if len(augmented['bboxes']) == 0:
                       annotation['bbox'] = [0,0,0,0]
                       annotation['class'] = 0
                   else:
                       annotation['bbox'] = augmented['bboxes'][0]
                       annotation['class'] = 1
               else:
                   annotation['bbox'] = [0,0,0,0]
                   annotation['class'] = 0
                   
               with open(os.path.join('aug_data', partition, 'labels', f'{image.split(".")[0]}.{x}.json'), 'w') as f:
                   json.dump(annotation, f)
                   
        except Exception as e:
            print(e)

#5.2 Load Augmented Images to Tensorflow Dataset

train_images = tf.data.Dataset.list_files('aug_data\\train\\images\\*.jpg', shuffle=False)
train_images = train_images.map(load_image)
train_images = train_images.map(lambda x: tf.image.resize(x, [120,120]))
train_images = train_images.map(lambda x: x/255)

test_images = tf.data.Dataset.list_files('aug_data\\test\\images\\*.jpg', shuffle=False)
test_images = test_images.map(load_image)
test_images = test_images.map(lambda x: tf.image.resize(x, [120,120]))
test_images = test_images.map(lambda x: x/255)

val_images = tf.data.Dataset.list_files('aug_data\\val\\images\\*.jpg', shuffle=False)
val_images = val_images.map(load_image)
val_images = val_images.map(lambda x: tf.image.resize(x, [120,120]))
val_images = val_images.map(lambda x: x/255)

#---------------------------------------------------------------------------------------------------------------------------------------------

#6 Prepare Labels

#6.1 Build Label Loading Function
def load_label(label_path):
    with open(label_path.numpy(), 'r', encoding = 'utf-8') as f:
        label = json.load(f)

    return [label['class']], label['bbox']

#6.2 Load Labels to Tensorflow Dataset

train_labels = tf.data.Dataset.list_files('aug_data\\train\\labels\\*.json', shuffle=False)
train_labels = train_labels.map(lambda x: tf.py_function(load_label, [x], [tf.uint8, tf.float16]))

test_labels = tf.data.Dataset.list_files('aug_data\\test\\labels\\*.json', shuffle=False)
test_labels = test_labels.map(lambda x: tf.py_function(load_label, [x], [tf.uint8, tf.float16]))

val_labels = tf.data.Dataset.list_files('aug_data\\val\\labels\\*.json', shuffle=False)
val_labels = val_labels.map(lambda x: tf.py_function(load_label, [x], [tf.uint8, tf.float16]))

#---------------------------------------------------------------------------------------------------------------------------------------------

#7 Combine Label and Image Samples

#7.1 Check Partition Lengths

len(train_images), len(train_labels), len(test_images), len(test_labels), len(val_images), len(val_labels)

#7.2 Create Final Dataset (Images/Labels)

train = tf.data.Dataset.zip((train_images, train_labels))
train = train.shuffle(5000)
train = train.batch(8)
train = train.prefetch(4)

test = tf.data.Dataset.zip((test_images, test_labels))
test = test.shuffle(5000)
test = test.batch(8)
test = test.prefetch(4)

val = tf.data.Dataset.zip((val_images, val_labels))
val = val.shuffle(5000)
val = val.batch(8)
val = val.prefetch(4)

#7.3 View Images and Annotations

data_samples = train.as_numpy_iterator()
# res = data_samples.next()

#---------------------------------------------------------------------------------------------------------------------------------------------

#8 Build Deep Learning using the Functional API

# 8.1 Import Layers and Base Network

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Dense, GlobalMaxPooling2D
from tensorflow.keras.applications import VGG16

#8.2 Download VGG16

vgg = VGG16(include_top=False)

#8.3 Build Instance of Network

def build_model():
    input_layer = Input(shape=(120,120,3)
                        )
    vgg = VGG16(include_top=False)(input_layer)
    
    f1 = GlobalMaxPooling2D()(vgg)
    class1 = Dense(2048, activation='relu')(f1)
    class2 = Dense(1, activation='sigmoid')(class1)
    
    f2 = GlobalMaxPooling2D()(vgg)
    regress1 = Dense(2048, activation='relu')(f2)
    regress2 = Dense(4, activation='sigmoid')(regress1)
    
    facetracker = Model(inputs=input_layer, outputs=[class2, regress2])
    return facetracker

#8.4 Test out Neural Network

facetracker = build_model()

X, y = train.as_numpy_iterator().next() 

classes, coords = facetracker.predict(X)

#---------------------------------------------------------------------------------------------------------------------------------------------

#9 Define Losses and Optimizers

#9.1 Define Opimizer and LR
batches_per_epoch = len(train)
lr_decay = (1./0.75 - 1)/batches_per_epoch

opt = tf.keras.optimizers.legacy.Adam(learning_rate=0.0001, decay=lr_decay)

#9.2 Create Localization Loss and Classification Loss
def localization_loss(y_true, yhat):
    delta_coord = tf.reduce_sum(tf.square(y_true[:,:2] - yhat[:,:2]))
    
    h_true = y_true[:,3] - y_true[:,1]
    w_true = y_true[:,2] - y_true[:,0]
    
    h_pred = yhat[:,3] - yhat[:,1]
    w_pred = yhat[:,2] - yhat[:,0]
    
    delta_size = tf.reduce_sum(tf.square(w_true - w_pred) + tf.square(h_true - h_pred))
    
    return delta_coord + delta_size

classloss = tf.keras.losses.BinaryCrossentropy()
regressloss = localization_loss

#---------------------------------------------------------------------------------------------------------------------------------------------

#10 Train Neural Network

#10.1 Createe Custom Model Class

class FaceTracker(Model):
    def __init__(self, eyetracker, **kwargs):
        super().__init__(**kwargs)
        self.model = eyetracker
        
    def compile(self, opt, classloss, localizationloss, **kwargs):
        super().compile(**kwargs)
        self.closs = classloss
        self.lloss = localizationloss
        self.opt = opt
    
    def train_step(self, batch, **kwargs):
      
        X, y = batch
        
        with tf.GradientTape() as tape:
            classes, coords = self.model(X, training=True)
            

            y0 = y[0]

            batch_classloss = self.closs(y0, classes)
            batch_localizationloss = self.lloss(tf.cast(y[1], tf.float32), coords)
            
            total_loss = batch_localizationloss + 0.5 * batch_classloss
            
            grad = tape.gradient(total_loss, self.model.trainable_variables)
            
        opt.apply_gradients(zip(grad, self.model.trainable_variables))
        
        return {'total_loss': total_loss, 'class_loss': batch_classloss, 'regress_loss': batch_localizationloss}
    
    def test_step(self, batch, **kwargs):
        X, y = batch
        
        classes, coords = self.model(X, training=False)
        
        batch_classloss = self.closs(y[0], classes)
        batch_localizationloss = self.lloss(tf.cast(y[1], tf.float32), coords)
        total_loss = batch_localizationloss + 0.5 * batch_classloss
        
        return {'total_loss': total_loss, 'class_loss': batch_classloss, 'regress_loss': batch_localizationloss}
    

model = FaceTracker(facetracker)
    
model.compile(opt, classloss, regressloss)

#10.2 Train Model

logdir = 'logs'

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

hist = model.fit(train, epochs=40, validation_data=val, callbacks=[tensorboard_callback])

#10.3 Plot Performance

fig, ax = plt.subplots(ncols=3, figsize=(20,5))

ax[0].plot(hist.history['total_loss'], color='teal', label='loss')
ax[0].plot(hist.history['val_total_loss'], color='orange', label='val loss')
ax[0].title.set_text('Loss')
ax[0].legend()

ax[1].plot(hist.history['class_loss'], color='teal', label='class loss')
ax[1].plot(hist.history['val_class_loss'], color='orange', label='val class loss')
ax[1].title.set_text('Classification Loss')
ax[1].legend()

ax[2].plot(hist.history['regress_loss'], color='teal', label='regress loss')
ax[2].plot(hist.history['val_regress_loss'], color='orange', label='val regress loss')
ax[2].title.set_text('Regression Loss')
ax[2].legend()

plt.show()

#---------------------------------------------------------------------------------------------------------------------------------------------

#11 Make Predictions

# 11.1 Make Predictions on Test Set

test_data = test.as_numpy_iterator()

test_sample = test_data.next()

yhat = facetracker.predict(test_sample[0])

fig, ax = plt.subplots(ncols=4, figsize=(20,20))
for idx in range(4):
    sample_image = test_sample[0][idx]
    newSample = sample_image.copy();
    sample_coords = yhat[1][idx]
    if yhat[0][idx] > 0.5:
        cv2.rectangle(newSample,
                      tuple(np.multiply(sample_coords[:2], [120,120]).astype(int)),
                      tuple(np.multiply(sample_coords[2:], [120,120]).astype(int)),
                      (255,0,0), 2)
    ax[idx].imshow(sample_image)
    
#11.2 Save the Model

from tensorflow.keras.models import load_model
facetracker.save('facetracker.h5')