import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt

# dataset file
datasetFileName = "pima-indians-diabetes.csv"

# initialize random number generator
seed = 7
numpy.random.seed(seed)

# load data
dataset = numpy.loadtxt(datasetFileName, delimiter=",")

# split dataset into input and output variables
X = dataset[:, 0:8]
Y = dataset[:, 8]

# define base model
def create_model():
    # create model
    model = Sequential()
    model.add(Dense(12, input_dim=8, kernel_initializer='uniform', activation='relu'))
    model.add(Dense(8, kernel_initializer='uniform', activation='relu'))
    model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))

    # compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# create the model
model = create_model()

history = model.fit(X, Y, validation_split=0.33, epochs=150, batch_size=10, verbose=0)

# evauate the model
scores = model.evaluate(X, Y)
print("\nAccuracy: %.2f%% \n" % (scores[1]*100))

# list all data in the history
print(history.history.keys())

# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# save the model as JSON file
# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

# save the weights
# serialized weight to HDF5
model.save_weights("model.h5")
print("\nSaved model to disk.")