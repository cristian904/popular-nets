from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Conv2D, Activation, MaxPooling2D, Flatten, Dense
from keras.optimizers import Adadelta
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

#prepare data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

img_rows = x_train[0].shape[0]
img_cols = x_train[1].shape[0]

x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype("float32")
x_test = x_test.astype("float32")

x_train /= 255
x_test /=255

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

num_classes = y_test.shape[1]
num_pixels = x_train.shape[1] * x_train.shape[2]

#recreate LeNet
model = Sequential()
model.add(Conv2D(20, (5, 5), padding="same", input_shape = input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(50, (5, 5), padding = "same"))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Flatten())
model.add(Dense(500))
model.add(Activation('relu'))

model.add(Dense(num_classes))
model.add(Activation('softmax'))

model.compile(loss = 'categorical_crossentropy',
                optimizer=Adadelta(), metrics=['accuracy'])

print(model.summary())

#callbacks
#save best parameter values in folder
checkpoint = ModelCheckpoint("./checkpoints",
                            monitor = "vall_loss",
                            mode = "min",
                            save_best_only=True,
                            verbose=1)

#stop if no improvements are seen in the val_loss
earlystop = EarlyStopping(monitor='val_loss',
                            min_delta=0,
                            patience=3,
                            verbose=1,
                            restore_best_weights=True)

#reduce learning rate if the val_loss has plateaued
reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                            factor=0.2,
                            patience=3,
                            verbose=1,
                            min_delta=0.0001)

callbacks = [checkpoint, earlystop, reduce_lr]

#training
batch_size = 128
epochs = 50

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_data=(x_test, y_test),
                    shuffle=True,
                    callbacks=callbacks)

#save model
model.save("./checkpoints/LeNet.h5")

#test_model
scores = model.evaluate(x_test, y_test, verbose=1)
print("Test accuracy:", scores[1])