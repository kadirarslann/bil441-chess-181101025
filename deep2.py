from __future__ import absolute_import, division, print_function, unicode_literals
import os

from evaluators import numbers

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

##############################################################################
def isNumber(s):
    return  numbers.__contains__(s)
def getNum(s):
    if s == "P":return 1
    if s == "R":return 4
    if s == "N":return 3
    if s == "B":return 2
    if s == "Q":return 8
    if s == "K":return 16
    if s == "p":return -1
    if s == "r":return -4
    if s == "n":return -3
    if s == "b":return -2
    if s == "q":return -8
    if s == "k":return -16

def fenTolist(fenString):
    Arr=[]
    a = np.zeros(shape=(64,))
    # print(a)
    columnindex = 0
    rowindex = 0
    for index, char in enumerate(fenString):
        if char == " ":
            break
        elif char == "/":
            pass
        elif isNumber(char):
            numval = int(char)
            while(numval!=0):
                a[columnindex]=0
                columnindex = columnindex + 1
                numval=numval-1
        else:
            a[columnindex]=getNum(char)
            columnindex = columnindex + 1
    return a
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
#################################################################################################################784906
def create_model(my_learning_rate):

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(64, input_shape=(64,)))
    model.add(tf.keras.layers.Dense(units=64,
                                    activation='relu',
                                    name='Hidden1'))
    model.add(tf.keras.layers.Dense(units=32,
                                    activation='relu',
                                    name='Hidden2'))
    model.add(tf.keras.layers.Dense(units=1,
                                    name='Output',
                                    activation='sigmoid'))
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=my_learning_rate),
                  loss="mean_squared_error",
                  metrics=[tf.keras.metrics.MeanSquaredError()])
    return model



print(train_df["Evaluation"][1088])
aaa = np.zeros(shape=(10000, 64))
bbb=    np.zeros(shape=(10000,))

for index, item in enumerate(aaa):
    if (index >= 10000):
        break
    aaa[index]=fenTolist(train_df["FEN"][index])
    if(train_df["Evaluation"][index][0]=="#"):
        if (train_df["Evaluation"][index][1]=="+"):
            bbb[index] = 4000 + (10-int(train_df["Evaluation"][index][2])) * 1000
            # print("beyaz avantaj", bbb[index])
        else:
            bbb[index] = -1 * (4000 + (10-int(train_df["Evaluation"][index][2])) * 1000)
            # print("siyah avantaj", bbb[index])
        # print(train_df["Evaluation"][index])
    else:
        bbb[index] = train_df["Evaluation"][index]


features=dict()
arr1=np.ndarray(shape=(2,2))
arr2=[]
for index, item in enumerate(aaa):
    arr1.append(item[33])
    arr2.append(item[43])

# for index, item in enumerate(aaa):
#     features[f'atr${index}']=aaa[index]
features["arr1"]=arr1
features["arr2"]=arr2
# for index, item in enumerate(aaa):
#     features[f'atr${index}']=aaa[index]

print(type(features),"---------")
print(type(features["arr1"]),"---------")
print(type(features["arr1"][1]),"---------")
def train_model(model, dataset, epochs, label_name,
                batch_size=None):
    # features={}
    #
    # features.appe
    # features = {name: np.array(value) for name, value in aaa.items()}

    history = model.fit(x=features, y=bbb, batch_size=batch_size,
                        epochs=epochs, shuffle=True)
    # The list of epochs is stored separately from the rest of history.
    epochs = history.epoch
    hist = pd.DataFrame(history.history)
    mse = hist["mean_squared_error"]
    return epochs,mse
# #################################################################################################################
learning_rate = 1
epochs = 15
batch_size = 100

# Specify the label
label_name = "Evaluation"

my_model = create_model(learning_rate)
# my_model.summary()
epochs,mse = train_model(my_model, train_df, epochs,
                          label_name, batch_size)
plot_the_loss_curve(epochs, mse)

board2d1=fenTolist('r1b1k2r/pp1n2pp/1qn1p3/3pp3/1b1P1P2/3B1N2/PP1BN1PP/R2QK2R w KQkq - 0 12') ##263
board2d2=fenTolist('r1b1k2r/pp1n2pp/1qn1p3/3pp3/1b1P1P2/3B1N2/PP1BN1PP/R2QK2R w KQkq - 0 12') ##915
board2d3=fenTolist('r3k2r/1b3ppp/pq2pn2/2b3B1/Pp6/3B1N1P/1PP1QPP1/R4RK1 b kq - 2 18') ## -56
# print(my_model.predict(board2d1)[0][0])
# print("------------------")
# print(my_model.predict(board2d2)[0][0])
# print("------------------")
# print(my_model.predict(board2d3)[0][0])

# After building a model against the training set, test that model
# against the test set.
# test_features = {name:np.array(value) for name, value in test_df.items()}
# test_label = np.array(test_features.pop(label_name)) # isolate the label
# print("\n Evaluate the new model against the test set:")
# my_model.evaluate(x = test_features, y = test_label, batch_size=batch_size)