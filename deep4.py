from __future__ import absolute_import, division, print_function, unicode_literals
import os

from algos import fenTolist,fenToBoard,fenToArrs,fenToArrs2,fenTolist2
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
# inputs = np.zeros(shape=(500000, 64))
inputs2 = np.zeros(shape=(100000, 256))
# inputs2d = np.zeros(shape=(50000, 8,8))
# inputs3d = np.zeros(shape=(50000, 14,8,8))
# inputsEnhanced=np.zeros(shape=(100000,4,64))
outputs = np.zeros(shape=(100000,))

for index, item in enumerate(inputs2):
    if (index >= 100000):
        break
    # inputs3d[index] = fenToArrs(train_df["FEN"][index])
    # inputs2d[index]=fenToBoard(train_df["FEN"][index])
    # inputs[index] = fenTolist(train_df["FEN"][index])
    inputs2[index] = fenTolist2(train_df["FEN"][index])
    # inputsEnhanced[index]= fenToArrs2(train_df["FEN"][index])
    if(train_df["Evaluation"][index][0]=="#"):
        if (train_df["Evaluation"][index][1]=="+"):
            outputs[index] = 3000 + (10-int(train_df["Evaluation"][index][2])) * 200
            # print("beyaz avantaj", bbb[index])
        else:
            outputs[index] = -1 * (3000 + (10-int(train_df["Evaluation"][index][2])) * 200)
            # print("siyah avantaj", bbb[index])
        # print(train_df["Evaluation"][index])
    else:
        outputs[index] = train_df["Evaluation"][index]

# std=outputs.std()
# mea=outputs.mean()
# outputs=(outputs-mea)/std
outputs=(outputs)/100
##############################################################################
def plot_the_loss_curve(epochs, mse):
  """Plot a curve of loss vs. epoch."""
  plt.figure()
  plt.xlabel("Epoch")
  plt.ylabel("Mean Squared Error")
  plt.plot(epochs, mse, label="Loss")
  plt.legend()
  plt.ylim([mse.min()*0.95, mse.max() * 1.00])
  plt.show()
print("Defined the plot_the_loss_curve function.")
#################################################################################################################

def create_model(my_learning_rate):
    """Create and compile a simple linear regression model."""
    # Most simple tf.keras models are sequential.
    model = tf.keras.models.Sequential()

    # model.add(tf.keras.layers.Dense(units=1,input_shape=(14,8,8 )))
    # model.add(tf.keras.layers.Dense(units=1,input_shape=(8,8)))
    # model.add(tf.keras.layers.Dense(units=1,input_shape=(4,64)))
    model.add(tf.keras.layers.Dense(units=8,input_shape=(256, )))

    # model.add(tf.keras.layers.Conv1D(filters=16, kernel_size=4, padding='same')),
    # model.add(tf.keras.layers.Dense(units=8))

    # model.add(tf.keras.layers.Conv1D(filters=32, kernel_size=3, padding='same', activation='softplus'))

    # model.add(tf.keras.layers.Dense(units=64))
    # model.add(tf.keras.layers.Dense(units=32 ))
    model.add(tf.keras.layers.Dense(units=16)) # best fot now list type
    model.add(tf.keras.layers.Dense(units=4)) # best fot now list type

    # Define the output layer.
    model.add(tf.keras.layers.Dense(units=1,name='Output'))
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=my_learning_rate),
                  loss="mean_squared_error",
                  metrics=[tf.keras.metrics.MeanSquaredError()])
    return model

def train_model(model, dataset, epochs, label_name,
                batch_size=None):
    # Split the dataset into features and label.
    # features = {name: np.array(value) for name, value in dataset.items()}
    # label = np.array(features.pop(label_name))
    history = model.fit(x=inputs2, y=outputs, batch_size=batch_size,validation_split=0.1,
                        epochs=epochs, shuffle=True)
    epochs = history.epoch
    hist = pd.DataFrame(history.history)
    mse = hist["mean_squared_error"]
    return epochs, mse
#################################################################################################################
learning_rate = 0.01 ############### best for now
epochs = 20
batch_size = 5000








# Establish the model's topography.
my_model = create_model(learning_rate)

epochs, mse = train_model(my_model, inputs2, epochs,
                          "FEN", batch_size)
plot_the_loss_curve(epochs, mse)





# myarr=fenToArrs("rnbqkb1r/pppppppp/8/4P3/3P4/4n3/PPP2PPP/RNBQKBNR w KQkq - 1 4")
# myarr2=fenToArrs("rnbqkb1r/ppp1pppp/B2p4/3nP3/3P4/8/PPP2PPP/RNBQK1NR b KQkq - 1 4")
# print(myarr)
# print("--------------------------------------------")
# myarr=np.expand_dims(myarr,0)
# myarr2=np.expand_dims(myarr2,0)
# print(my_model.predict(myarr)[0][0][0])
# print(my_model.predict(myarr2)[0][0][0])

myarr=fenTolist2("rnbqkb1r/pppppppp/5n2/7Q/4P3/8/PPPP1PPP/RNB1KBNR b KQkq - 2 2") #-1200
myarr2=fenTolist2("rnbqkb1r/pppppppp/B4n2/8/4P3/8/PPPP1PPP/RNBQK1NR b KQkq - 2 2") # -400
myarr3=fenTolist2("rnbqkb1r/pppppppp/8/3nP3/8/8/PPPPKPPP/RNBQ1BNR b kq - 2 3") #-230
myarr4=fenTolist2("rnbqkb1r/pppppppp/8/4P3/4n3/8/PPPP1PPP/RNBQKBNR w KQkq - 1 3")  #184
myarr5=fenTolist2("rnbqkb1r/pppppp1p/5n2/4P1p1/8/8/PPPP1PPP/RNBQKBNR w KQkq - 0 3") #412
myarr6=fenTolist2("rn1qkb1r/ppp1pppp/1n1P4/8/2PP4/7b/PP3PPP/RNBQKBNR w KQkq - 1 6")  #625
myarr=np.expand_dims(myarr,0)
myarr2=np.expand_dims(myarr2,0)
myarr3=np.expand_dims(myarr3,0)
myarr4=np.expand_dims(myarr4,0)
myarr5=np.expand_dims(myarr5,0)
myarr6=np.expand_dims(myarr6,0)
print(my_model.predict(myarr)[0][0])
print(my_model.predict(myarr2)[0][0])
print(my_model.predict(myarr3)[0][0])
print(my_model.predict(myarr4)[0][0])
print(my_model.predict(myarr5)[0][0])
print(my_model.predict(myarr6)[0][0])

# myarr=fenToBoard("rnbqkb1r/pppppppp/5n2/7Q/4P3/8/PPPP1PPP/RNB1KBNR b KQkq - 2 2") #-1200
# myarr2=fenToBoard("rnbqkb1r/pppppppp/B4n2/8/4P3/8/PPPP1PPP/RNBQK1NR b KQkq - 2 2") # -400
# myarr3=fenToBoard("rnbqkb1r/pppppppp/8/3nP3/8/8/PPPPKPPP/RNBQ1BNR b kq - 2 3") #-230
# myarr4=fenToBoard("rnbqkb1r/pppppppp/8/4P3/4n3/8/PPPP1PPP/RNBQKBNR w KQkq - 1 3")  #184
# myarr5=fenToBoard("rnbqkb1r/pppppp1p/5n2/4P1p1/8/8/PPPP1PPP/RNBQKBNR w KQkq - 0 3") #412
# myarr6=fenToBoard("rn1qkb1r/ppp1pppp/1n1P4/8/2PP4/7b/PP3PPP/RNBQKBNR w KQkq - 1 6")  #625
# myarr=np.expand_dims(myarr,0)
# myarr2=np.expand_dims(myarr2,0)
# myarr3=np.expand_dims(myarr3,0)
# myarr4=np.expand_dims(myarr4,0)
# myarr5=np.expand_dims(myarr5,0)
# myarr6=np.expand_dims(myarr6,0)
# print(my_model.predict(myarr)[0][0])
# print(my_model.predict(myarr2)[0][0])
# print(my_model.predict(myarr3)[0][0])
# print(my_model.predict(myarr4)[0][0])
# print(my_model.predict(myarr5)[0][0])
# print(my_model.predict(myarr6)[0][0])

# myarr=fenToArrs("rnbqkb1r/pppppppp/5n2/7Q/4P3/8/PPPP1PPP/RNB1KBNR b KQkq - 2 2") #-1200
# myarr2=fenToArrs("rnbqkb1r/pppppppp/B4n2/8/4P3/8/PPPP1PPP/RNBQK1NR b KQkq - 2 2") # -400
# myarr3=fenToArrs("rnbqkb1r/pppppppp/8/3nP3/8/8/PPPPKPPP/RNBQ1BNR b kq - 2 3") #-230
# myarr4=fenToArrs("rnbqkb1r/pppppppp/8/4P3/4n3/8/PPPP1PPP/RNBQKBNR w KQkq - 1 3")  #184
# myarr5=fenToArrs("rnbqkb1r/pppppp1p/5n2/4P1p1/8/8/PPPP1PPP/RNBQKBNR w KQkq - 0 3") #412
# myarr6=fenToArrs("rn1qkb1r/ppp1pppp/1n1P4/8/2PP4/7b/PP3PPP/RNBQKBNR w KQkq - 1 6")  #625
# myarr=np.expand_dims(myarr,0)
# myarr2=np.expand_dims(myarr2,0)
# myarr3=np.expand_dims(myarr3,0)
# myarr4=np.expand_dims(myarr4,0)
# myarr5=np.expand_dims(myarr5,0)
# myarr6=np.expand_dims(myarr6,0)
# print(my_model.predict(myarr)[0][0])
# print(my_model.predict(myarr2)[0][0])
# print(my_model.predict(myarr3)[0][0])
# print(my_model.predict(myarr4)[0][0])
# print(my_model.predict(myarr5)[0][0])
# print(my_model.predict(myarr6)[0][0])

# myarr=fenToArrs2("rnbqkb1r/pppppppp/5n2/7Q/4P3/8/PPPP1PPP/RNB1KBNR b KQkq - 2 2") #-1200
# myarr2=fenToArrs2("rnbqkb1r/pppppppp/B4n2/8/4P3/8/PPPP1PPP/RNBQK1NR b KQkq - 2 2") # -400
# myarr3=fenToArrs2("rnbqkb1r/pppppppp/8/3nP3/8/8/PPPPKPPP/RNBQ1BNR b kq - 2 3") #-230
# myarr4=fenToArrs2("rnbqkb1r/pppppppp/8/4P3/4n3/8/PPPP1PPP/RNBQKBNR w KQkq - 1 3")  #184
# myarr5=fenToArrs2("rnbqkb1r/pppppp1p/5n2/4P1p1/8/8/PPPP1PPP/RNBQKBNR w KQkq - 0 3") #412
# myarr6=fenToArrs2("rn1qkb1r/ppp1pppp/1n1P4/8/2PP4/7b/PP3PPP/RNBQKBNR w KQkq - 1 6")  #625
# myarr=np.expand_dims(myarr,0)
# myarr2=np.expand_dims(myarr2,0)
# myarr3=np.expand_dims(myarr3,0)
# myarr4=np.expand_dims(myarr4,0)
# myarr5=np.expand_dims(myarr5,0)
# myarr6=np.expand_dims(myarr6,0)
# print(my_model.predict(myarr)[0][0])
# print(my_model.predict(myarr2)[0][0])
# print(my_model.predict(myarr3)[0][0])
# print(my_model.predict(myarr4)[0][0])
# print(my_model.predict(myarr5)[0][0])
# print(my_model.predict(myarr6)[0][0])