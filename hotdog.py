import numpy as np
import os
from keras.preprocessing.image import load_img, img_to_array
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.resnet50 import preprocess_input
from keras import Sequential
from keras.layers import Flatten, Dense, Conv2D, Dropout, BatchNormalization, Activation, MaxPooling2D
from keras import optimizers, regularizers
from keras.applications import VGG16

train_path = './data/train'
test_path = './data/test'
hot_dog_path = './data/train/hot_dog'
not_hot_dog_path = './data/train/not_hot_dog'

train_data_hd = [os.path.join(hot_dog_path, filename)\
                 for filename in os.listdir(hot_dog_path)]
train_data_nhd = [os.path.join(not_hot_dog_path, filename)\
                  for filename in os.listdir(not_hot_dog_path)]

img_size = 150 #224
num_classes = 2 # hot dog or not hot dog

data_generator = ImageDataGenerator(rescale=1./255,
				rotation_range=90,
				width_shift_range=0.2,
				height_shift_range=0.2,
				horizontal_flip=True,
				shear_range=0.2,
				zoom_range=0.2)

test_generator = ImageDataGenerator(rescale=1./255)

train_generator = data_generator.flow_from_directory(
    train_path,
    target_size=(img_size, img_size),
    batch_size=498, #498
    class_mode='binary') #categorical

validation_generator = test_generator.flow_from_directory(
    test_path,
    target_size=(img_size, img_size),
    batch_size=500, #500
    class_mode='binary') #categorical


def read_and_prep_images(img_paths, img_height=img_size, img_width=img_size):
    imgs = [load_img(img_path, target_size=(img_height, img_width)) for img_path in img_paths]
    img_array = np.array([img_to_array(img) for img in imgs])
    output = preprocess_input(img_array)
    return output


train_data = read_and_prep_images(train_data_nhd)
train_data1 = read_and_prep_images(train_data_hd)

conv_base = VGG16(weights='imagenet',
			include_top=False,
			input_shape=(img_size, img_size, 3))
conv_base.trainable = False

model = Sequential()
model.add(conv_base)
'''
model.add(Conv2D(16, (3,3), activation='relu', input_shape=(img_size, img_size, 3)))
model.add(MaxPooling2D((2,2)))
model.add(Conv2D(32, (3,3), activation='relu'))
model.add(MaxPooling2D((2,2)))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D((2,2)))
model.add(Conv2D(128, (3,3), activation='relu'))
model.add(MaxPooling2D((2,2)))
'''

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.summary()
#LEARNING RATE
learning_rate = 1E-4
model.compile(loss='binary_crossentropy',
	      optimizer=optimizers.RMSprop(lr=learning_rate),
	      metrics=['accuracy'])

input_x = (train_generator[0][0])
input_y = (train_generator[0][1])

model.fit(input_x,
          input_y,
          batch_size=32,
	  validation_split=0.1,
          epochs=5)

output_x = (validation_generator[0][0])
output_y = validation_generator[0][1]

loss_and_acc = model.evaluate(output_x, output_y)
print('loss = ' + str(loss_and_acc[0]))
print('accuracy = ' + str(loss_and_acc[1]))
