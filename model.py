import os
import cv2
import numpy as np
import csv
import tensorflow

from keras.models import Sequential
from keras.layers import Flatten, Dense, Conv2D, MaxPooling2D 
from keras.layers import Lambda, Cropping2D, Dropout, ELU
from keras.layers import BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

#load lines in csv file to read the images 
def load_csv(file):
    lines = []
    with open(file) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)
    return lines

file = 'data/driving_log.csv'
image_path = 'data/IMG/'

#load image names in the csv file
lines_path = load_csv(file)

#split not the data just their names
# We don't need to test images because it is a regression problem not classification.
train_samples, validation_samples = train_test_split(lines_path, shuffle=True, test_size=0.2)

print('done')

def generator(samples, batch_size=32):
  num_samples = len(samples)
  while 1:
    shuffle(samples)
    for offset in range(0, num_samples, batch_size):
      batch_samples = samples[offset:offset+batch_size]

      images = []
      measurements = []
      for batch_sample in batch_samples:
        corrections = [0, 0.2, -0.2] # center, left, right
        # The following loop takes data from three cameras: center, left, and right.
        # The steering measurement for each camera is then added by
        # the correction as listed above.
        for i, c in enumerate(corrections):
          source_path = batch_sample[i]
          filename = source_path.split('/')[-1]
          current_path = os.path.join(image_path, os.path.basename(filename))

          image = cv2.imread(current_path)
          image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
          image = np.asarray(image)

          images.append(image)
          measurement = float(batch_sample[3]) + c
          measurements.append(measurement)

          # Flip
          image_flipped = np.fliplr(image)
          images.append(image_flipped)
          measurement_flipped = -measurement
          measurements.append(measurement_flipped)

      X_train = np.array(images)
      y_train = np.array(measurements)
      yield shuffle(X_train, y_train)

# Load the images via generator
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)
print('done')


def model_config():
  model = Sequential()

# Crop 75 pixels from the top of the image and 25 from the bottom
  model.add(Cropping2D(cropping=((75, 25), (0, 0)),
                     input_shape=(160, 320, 3),
                     data_format="channels_last"))

# Normalize the data
  model.add(Lambda(lambda x: (x/127.5) - 0.5))

  model.add(Conv2D(3, (1, 1), padding='same'))
  model.add(ELU())

  model.add(BatchNormalization())
  model.add(Conv2D(16, (5, 5), strides=(2, 2), padding="same"))
  model.add(ELU())

  model.add(BatchNormalization())
  model.add(Conv2D(32, (5, 5), strides=(2, 2), padding="same"))
  model.add(ELU())

  model.add(BatchNormalization())
  model.add(Conv2D(64, (3, 3), strides=(2, 2), padding="same"))
  model.add(ELU())

  model.add(BatchNormalization())
  model.add(Conv2D(128, (3, 3), strides=(2, 2), padding="same"))
  model.add(ELU())

  model.add(Flatten())
  model.add(ELU())

  model.add(Dense(512))
  model.add(Dropout(.2))
  model.add(ELU())

  model.add(Dense(100))
  model.add(Dropout(.5))
  model.add(ELU())

  model.add(Dense(10))
  model.add(Dropout(.5))
  model.add(ELU())

  model.add(Dense(1))

  adam = Adam(lr=1e-5)
  model.compile(optimizer= adam, loss="mse", metrics=['accuracy'])

  model.summary()

  return model

file = 'model.h5'


model = model_config()
earlystopper = EarlyStopping(patience=5, verbose=1)
checkpointer = ModelCheckpoint(file, monitor='val_loss', verbose=1, save_best_only=True)
model.fit_generator(train_generator, steps_per_epoch= len(train_samples),
                    validation_data=validation_generator,
                    validation_steps=len(validation_samples), epochs = 5)
model.save(file)
print('done')

# Plotting
print(history_object.history.keys())
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()


