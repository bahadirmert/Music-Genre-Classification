import configparser

from keras.callbacks import ModelCheckpoint
from keras import optimizers
from keras import losses
from keras.utils import np_utils
from os.path import isfile

from Extract_Audio_Features import ExtractAudioFeatures
from Get_Train_Test_Data import GetTrainTestData
from Neural_Network import CNNModel

config = configparser.ConfigParser()
config.read('config.ini')


ExtractAudioFeatures(config).prepossessingAudio()

X_train, X_test, y_train, y_test = GetTrainTestData(config).split_dataset()

X_train = X_train.reshape(X_train.shape[0], 128, 625, 1).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 128, 625, 1).astype('float32')

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]

print("Working..")

model = CNNModel(config, X_train).build_model(nb_classes = num_classes)

model.compile(loss = losses.categorical_crossentropy,
              optimizer = optimizers.SGD(lr=0.001, momentum=0, decay=1e-5, nesterov=True),
              metrics = ['accuracy'])
model.summary()

if int(config['CNN_CONFIGURATION']['LOAD_CHECKPOINT']):
    print("Looking for previous weights...")
    if isfile(config['CNN_CONFIGURATION']['CHECKPOINT_PATH']):
        print('Checkpoint file detected. Loading weights.')
        model.load_weights(config['CNN_CONFIGURATION']['CHECKPOINT_PATH'])
    else:
        print('No checkpoint file detected.  Starting from scratch.')
else:
    print('Starting from scratch (no checkpoint)')

checkpointer = ModelCheckpoint(filepath = config['CNN_CONFIGURATION']['CHECKPOINT_PATH'], verbose = 1, save_best_only = True)

model.fit(
    X_train,
    y_train,
    batch_size = int(config['CNN_CONFIGURATION']['BATCH_SIZE']),
    epochs = int(config['CNN_CONFIGURATION']['NUMBERS_EPOCH']),
    verbose = 1,
    validation_data = (X_test, y_test),
    callbacks = [checkpointer])

score = model.evaluate(X_test, y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])
model.save_weights('/output/weights.hdf5')