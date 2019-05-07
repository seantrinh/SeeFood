import numpy as np
import os
from keras.preprocessing.image import load_img, img_to_array
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.resnet50 import preprocess_input
from keras import Sequential
from keras.layers import Flatten, Dense, Conv2D, Dropout, BatchNormalization, Activation, MaxPooling2D
from keras import optimizers

train_path = './data/train'
test_path = './data/test'
hot_dog_path = './data/train/hot_dog'
not_hot_dog_path = './data/train/not_hot_dog'

train_data_hd = [os.path.join(hot_dog_path, filename)\
                 for filename in os.listdir(hot_dog_path)]
train_data_nhd = [os.path.join(not_hot_dog_path, filename)\
                  for filename in os.listdir(not_hot_dog_path)]

img_size = 224
num_classes = 2 # hot dog or not hot dog

data_generator = ImageDataGenerator(rotation_range=90, width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)

train_generator = data_generator.flow_from_directory(
    train_path,
    target_size=(img_size, img_size),
    batch_size=498,
    class_mode='categorical')

validation_generator = data_generator.flow_from_directory(
    test_path,
    target_size=(img_size, img_size),
    batch_size=500,
    class_mode='categorical')


def read_and_prep_images(img_paths, img_height=img_size, img_width=img_size):
    imgs = [load_img(img_path, target_size=(img_height, img_width)) for img_path in img_paths]
    img_array = np.array([img_to_array(img) for img in imgs])
    output = preprocess_input(img_array)
    return output


train_data = read_and_prep_images(train_data_nhd)
train_data1 = read_and_prep_images(train_data_hd)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3,3), input_shape=(img_size, img_size, 3)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D((2,2)))
model.add(Conv2D(64, kernel_size=(3,3)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D((2,2)))
model.add(Conv2D(128, kernel_size=(3,3)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D((2,2)))
model.add(Conv2D(256, kernel_size=(3,3)))
model.add(BatchNormalization())
model.add(Activation('relu'))
#model.add(Conv2D(32, kernel_size=(3,3)))
#model.add(BatchNormalization())
#model.add(Activation('relu'))

model.add(Flatten())
model.add(Dropout(0.3))
model.add(Dense(128))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dense(num_classes, activation='softmax'))

model.summary()
#LEARNING RATE
learning_rate = 1E-4
model.compile(loss='binary_crossentropy',
              optimizer=optimizers.Adam(lr=learning_rate),
              metrics=['accuracy'])

input_x = (train_generator[0][0]/255)
input_y = (train_generator[0][1])

model.fit(input_x,
          input_y,
          batch_size=24,
          epochs=10)

output_x = (validation_generator[0][0]/255)
output_y = validation_generator[0][1]

loss_and_acc = model.evaluate(output_x, output_y)
print('loss = ' + str(loss_and_acc[0]))
print('accuracy = ' + str(loss_and_acc[1]))
