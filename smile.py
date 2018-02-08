import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten, Reshape
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
from wandb.wandb_keras import WandbKerasCallback
from skimage import filters
import wandb
import smiledataset

run = wandb.init()
config = run.config

config.epochs=10

# load data
train_X, train_y, test_X, test_y = smiledataset.load_data()

train_X /= 255.0
test_X /= 255.0

# data augmentation
# flip x-y
flip_train_X = np.array(train_X)
for img in flip_train_X:
	img = img[:, ::-1]

np.append(flip_train_X, train_X)
np.append(train_y, train_y)

# blur
blur_train_X = np.array(train_X)
for img in blur_train_X:
	img = filters.gaussian(img, sigma=16)

np.append(blur_train_X, train_X)
np.append(train_y, train_y)

# shuffle?

# convert classes to vector
num_classes = 2
train_y = np_utils.to_categorical(train_y, num_classes)
test_y = np_utils.to_categorical(test_y, num_classes)

img_rows, img_cols = train_X.shape[1:]

# add additional dimension
test_X = test_X.reshape(test_X.shape[0], test_X.shape[1], test_X.shape[2], 1)
train_X = train_X.reshape(train_X.shape[0], train_X.shape[1], train_X.shape[2], 1)

model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape=(img_rows, img_cols,1), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

#model.add(Conv2D(32, (3, 3), activation='relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))

#model.add(Conv2D(64, (3, 3), activation='relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(num_classes, activation='softmax') )

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(train_X, train_y,
    epochs=config.epochs, verbose=1,
    validation_data=(test_X, test_y), callbacks=[WandbKerasCallback()])

model.save("smile.h5")
