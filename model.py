import csv
from PIL import Image
import cv2
import numpy as np
import sklearn
from random import shuffle

MODEL_FILE = 'model.h5'
LOSS_PLOT_FILE = 'loss_plot.png'
STEERING_CORRECTION = 0.2
IMG_PATH = 'data/IMG/'
EPOCHS = 5
BATCH_SIZE = 32
samples = []

def image_path(source_path):
    filename = source_path.split('/')[-1]
    return IMG_PATH + filename

def process_image(source_path):
    image = Image.open(image_path(source_path))
    return np.asarray(image)

with open('data/driving_log.csv') as csvfile:
    reader  = csv.reader(csvfile)
    for sample in reader:
        samples.append(sample)
    samples = samples[1:] # Skip header row

images = []
measurements = []
for batch_sample in samples:
    steering_center = float(batch_sample[3])
    steering_left   = steering_center + STEERING_CORRECTION
    steering_right  = steering_center - STEERING_CORRECTION

    img_center = process_image(batch_sample[0])
    img_left   = process_image(batch_sample[1])
    img_right  = process_image(batch_sample[2])

    images += [img_center, img_left, img_right]
    measurements += [steering_center, steering_left, steering_right]

augmented_images, augmented_measurements = [], []
for image,measurement in zip(images, measurements):
    image_flipped = np.fliplr(image)
    measurement_flipped = -measurement

    augmented_images += [image, image_flipped]
    augmented_measurements += [measurement, measurement_flipped]

X_train = np.array(images)
y_train = np.array(measurements)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Cropping2D, Lambda
import matplotlib.pyplot as plt
plt.switch_backend('agg')

model = Sequential()
model.add(Cropping2D(cropping=((70,25), (0,0)), input_shape=(160,320,3)))
model.add(Lambda(lambda x: (x / 255.0) - 0.5))
model.add(Flatten())
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')

history_object = model.fit(X_train,
                        y_train,
                        validation_split=0.2,
                        shuffle=True)

model.save(MODEL_FILE)

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.savefig(LOSS_PLOT_FILE)
