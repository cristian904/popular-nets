#building a monkey classifier
from keras.applications import MobileNet
from keras.models import Model
from keras.layers import GlobalAveragePooling2D, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import RMSprop
from keras.callbacks import EarlyStopping, ModelCheckpoint

#Mobile net works on 224 x 224 images
img_rows, img_cols = 224, 224

MobileNet = MobileNet(weights= 'imagenet',
                    include_top = False, # don't include FC layers
                    input_shape = (img_rows, img_cols, 3))

for layer in MobileNet.layers:
    layer.trainable = False

def add_top_model_mobilenet(bottom_model, num_classes):
    top_model = bottom_model.output
    top_model = GlobalAveragePooling2D()(top_model)
    top_model = Dense(1024, activation='relu')(top_model)
    top_model = Dense(1024, activation='relu')(top_model)
    top_model = Dense(512, activation='relu')(top_model)
    top_model = Dense(num_classes, activation='softmax')(top_model)
    return top_model

num_classes = 10

FC_Head = add_top_model_mobilenet(MobileNet, num_classes)
model = Model(inputs = MobileNet.inputs, outputs = FC_Head)

train_data_dir = "./monkey_breed/train"
validation_data_dir = "./monkey_breed/validation"

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=45,
    width_shift_range=0.3,
    height_shift_range=0.3,
    horizontal_flip=True,
    fill_mode='nearest')

validation_datagen = ImageDataGenerator(rescale=1./255)

batch_size = 32

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_rows, img_cols),
    batch_size=batch_size,
    class_mode='categorical'
)

validation_generator = validation_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_rows, img_cols),
    batch_size=batch_size,
    class_mode='categorical'
)

checkpoint = ModelCheckpoint(
    "../checkpoints/mobileNet.h5",
    monitor='val_loss',
    mode='min',
    save_best_only=True,
    verbose=1
)

earlystop = EarlyStopping(
    monitor='val_loss',
    min_delta=0,
    patience=3,
    verbose=1,
    restore_best_weights=True
)

callbacks = [checkpoint, earlystop]

model.compile(
    loss='categorical_crossentropy',
    optimizer = RMSprop(lr=0.001),
    metrics = ['accuracy']
)

nb_train_samples = 1097
nb_validation_samples = 272

epochs = 5
batch_size = 16

history = model.fit_generator(
    train_generator,
    steps_per_epoch = nb_train_samples // batch_size,
    epochs = epochs,
    callbacks = callbacks,
    validation_data=validation_generator,
    validation_steps = nb_validation_samples // batch_size
)
