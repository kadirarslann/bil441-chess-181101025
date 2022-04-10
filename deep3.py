from __future__ import absolute_import, division, print_function, unicode_literals
import os

from algos import fenTolist,fenToBoard

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
from matplotlib import pyplot as plt
import seaborn as sns

import warnings

warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

pd.options.display.max_rows = 10
pd.options.display.float_format = "{:.1f}".format

print("Imported modules.")
##############################################################################
train_df = pd.read_csv("./samples/random_evals.csv")
train_df = train_df.reindex(np.random.permutation(train_df.index)) # shuffle the examples
test_df = pd.read_csv("./samples/random_evals.csv")


##############################################################################
inputs = np.zeros(shape=(1000000, 64))
# inputs2d = np.zeros(shape=(100000, 8,8))
# inputs1 = np.zeros(shape=(100000,))
outputs = np.zeros(shape=(1000000,))

for index, item in enumerate(inputs):
    if (index >= 1000000):
        break
    # inputs2d[index]=fenToBoard(train_df["FEN"][index])
    inputs[index] = fenTolist(train_df["FEN"][index])
    if(train_df["Evaluation"][index][0]=="#"):
        if (train_df["Evaluation"][index][1]=="+"):
            outputs[index] = 4000 + (10-int(train_df["Evaluation"][index][2])) * 500
            # print("beyaz avantaj", bbb[index])
        else:
            outputs[index] = -1 * (4000 + (10-int(train_df["Evaluation"][index][2])) * 500)
            # print("siyah avantaj", bbb[index])
        # print(train_df["Evaluation"][index])
    else:
        outputs[index] = train_df["Evaluation"][index]
print(inputs[1])
print(type(inputs[1]))
print(type(inputs))
##############################################################################
def plot_the_loss_curve(epochs, mse):
  """Plot a curve of loss vs. epoch."""
  plt.figure()
  plt.xlabel("Epoch")
  plt.ylabel("Mean Squared Error")
  plt.plot(epochs, mse, label="Loss")
  plt.legend()
  plt.ylim([mse.min()*0.95, mse.max() * 1.03])
  plt.show()
print("Defined the plot_the_loss_curve function.")
#################################################################################################################

#################################################################################################################
def create_model(my_learning_rate):
    """Create and compile a simple linear regression model."""
    # Most simple tf.keras models are sequential.
    model = tf.keras.models.Sequential()
    # Add the layer containing the feature columns to the model.
    # model.add(my_feature_layer)
    # model.add(tf.keras.layers.Dense(units=1,
    #                                 input_shape=(8,8)))
    model.add(tf.keras.layers.Dense(units=1,
                                    input_shape=(64, )))
    # Define the first hidden layer with 20 nodes.
    model.add(tf.keras.layers.Dense(units=64,activation='relu',name='Hidden1'))
    # Define the second hidden layer with 12 nodes.
    model.add(tf.keras.layers.Dense(units=16,activation='relu',name='Hidden2'))
    # Define the second hidden layer with 12 nodes.
    model.add(tf.keras.layers.Dense(units=4,activation='relu',name='Hidden3'))
    # Define the output layer.
    model.add(tf.keras.layers.Dense(units=1,name='Output'))
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=my_learning_rate),
                  loss="mean_squared_error",
                  metrics=[tf.keras.metrics.MeanSquaredError()])
    return model

def train_model(model, dataset, epochs, label_name,
                batch_size=None):
    """Train the model by feeding it data."""
    # Split the dataset into features and label.
    # features = {name: np.array(value) for name, value in dataset.items()}
    # label = np.array(features.pop(label_name))
    history = model.fit(x=inputs, y=outputs, batch_size=batch_size,validation_split=0.1,
                        epochs=epochs, shuffle=True)
    # The list of epochs is stored separately from the rest of history.
    epochs = history.epoch
    # To track the progression of training, gather a snapshot
    # of the model's mean squared error at each epoch.
    hist = pd.DataFrame(history.history)
    mse = hist["mean_squared_error"]
    return epochs, mse
#################################################################################################################
learning_rate = 0.2
epochs = 50
batch_size = 10000

# Specify the label

# Establish the model's topography.
my_model = create_model(learning_rate)
# Train the model on the normalized training set. We're passing the entire
# normalized training set, but the model will only use the features
# defined by the feature_layer.
epochs, mse = train_model(my_model, inputs, epochs,
                          "FEN", batch_size)
plot_the_loss_curve(epochs, mse)
myarr=fenTolist("rnbqkb1r/pppppppp/5n2/7Q/4P3/8/PPPP1PPP/RNB1KBNR b KQkq - 2 2")
print(myarr)
myarr=np.expand_dims(myarr,0)
print(my_model.predict(myarr)[0][0])
# After building a model against the training set, test that model
# against the test set.
# test_features = {name:np.array(value) for name, value in test_df_norm.items()}
# test_label = np.array(test_features.pop(label_name)) # isolate the label
# print("\n Evaluate the new model against the test set:")
# my_model.evaluate(x = myarr, y = test_label, batch_size=1)