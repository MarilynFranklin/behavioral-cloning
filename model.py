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
EPOCHS = 10
BATCH_SIZE = 32
samples = []

def image_path(source_path):
    filename = source_path.split('/')[-1]
    return IMG_PATH + filename

def process_image(source_path):
    image = Image.open(image_path(source_path))
    image_array = np.asarray(image)[:,:,1]
    return np.expand_dims(image_array, axis=2)

with open('data/driving_log.csv') as csvfile:
    reader  = csv.reader(csvfile)
    for sample in reader:
        samples.append(sample)
    samples = samples[1:] # Skip header row

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            measurements = []
            for batch_sample in batch_samples:
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

            X_train = np.array(augmented_images)
            y_train = np.array(augmented_measurements)

            yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=BATCH_SIZE)
validation_generator = generator(validation_samples, batch_size=BATCH_SIZE)

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda, ELU, Cropping2D
from keras.layers.convolutional import Convolution2D
import matplotlib.pyplot as plt
plt.switch_backend('agg')

def get_model():
    ch, row, col = 1, 160, 320  # camera format

    model = Sequential()
    model.add(Cropping2D(cropping=((70,25), (0,0)), input_shape=(160,320,1)))
    model.add(Lambda(lambda x: (x / 127.5) - 1.))
    model.add(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode="same"))
    model.add(ELU())
    model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(ELU())
    model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(Flatten())
    model.add(Dropout(.2))
    model.add(ELU())
    model.add(Dense(512))
    model.add(Dropout(.5))
    model.add(ELU())
    model.add(Dense(1))

    model.compile(optimizer="adam", loss="mse")

    return model

model = get_model()

history_object = model.fit_generator(train_generator,
                        samples_per_epoch=len(train_samples),
                        validation_data=validation_generator,
                        nb_val_samples=len(validation_samples),
                        nb_epoch=EPOCHS)

model.save(MODEL_FILE)

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.savefig(LOSS_PLOT_FILE)
