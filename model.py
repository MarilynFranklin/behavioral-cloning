import csv
import cv2
import numpy as np

MODEL_FILE = 'model.h5'
LOSS_PLOT_FILE = 'loss_plot.png'
STEERING_CORRECTION = 0.2
IMG_PATH = 'data/IMG/'
rows = []

def image_path(source_path)
    filename = source_path.split('/')[-1]
    IMG_PATH + filename

images = []
measurements = []

with open('data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    rows   = reader[1:] # Skip header row

for row in rows:
    steering_center = float(row[3])
    steering_left   = steering_center + STEERING_CORRECTION
    steering_right  = steering_center - STEERING_CORRECTION

    img_center = cv2.imread(image_path(row[0]))
    img_left   = cv2.imread(image_path(row[1]))
    img_right  = cv2.imread(image_path(row[2]))

    images.extend(img_center, img_left, img_right)
    measurements.extend(steering_center, steering_left, steering_right)

X_train = np.array(images)
y_train = np.array(measurements)

from keras.models import Sequential
from keras.layers import Flatten, Dense
import matplotlib.pyplot as plt
plt.switch_backend('agg')

model = Sequential()
model.add(Flatten(input_shape=(160,320,3)))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
history_object = model.fit(X_train, y_train, validation_split=0.2, shuffle=True)

model.save(MODEL_FILE)

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.savefig(LOSS_PLOT_FILE)
