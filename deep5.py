from __future__ import absolute_import, division, print_function, unicode_literals
import os

from algos import fenTolist,fenToBoard,fenToArrs,fenToArrs2,fenTolist2
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models
import tensorflow.keras.callbacks as callbacks
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
            outputs[index] = 3000 + (10-int(train_df["Evaluation"][index][2])) * 100
            # print("beyaz avantaj", bbb[index])
        else:
            outputs[index] = -1 * (3000 + (10-int(train_df["Evaluation"][index][2])) * 100)
            # print("siyah avantaj", bbb[index])
        # print(train_df["Evaluation"][index])
    else:
        outputs[index] = train_df["Evaluation"][index]

# std=outputs.std()
# mea=outputs.mean()
# outputs=(outputs-mea)/std
outputs=outputs/1000;
##############################################################################
def plot_the_loss_curve(epochs, mse):
  """Plot a curve of loss vs. epoch."""
  plt.figure()
  plt.xlabel("Epoch")
  plt.ylabel("Mean Squared Error")
  plt.plot(epochs, mse, label="Loss")
  plt.legend()
  plt.ylim([mse.min(), mse.max() * 1.00])
  plt.show()
print("Defined the plot_the_loss_curve function.")
#################################################################################################################

def create_model():
    board1d = layers.Input(shape=(256,))
    x = board1d

    # for _ in range(2): #best so far
    #     x = layers.Dense(64, activation="relu")(x)
    # x = layers.Flatten()(x)
    # x = layers.Dense(64, 'relu')(x)
    # x = layers.Dense(1, 'sigmoid')(x)

    # for _ in range(2): #best so far-2
    #     x = layers.Dense(64,activation="softplus")(x)
    # # x = layers.Flatten()(x)
    # x = layers.Dense(64,activation="softplus")(x)
    # x = layers.Dense(1,activation="sigmoid")(x)
    # return models.Model(inputs=board3d, outputs=x)

    x = layers.Dense(64, activation="relu")(x)  # best so far-3 --------
    # x = layers.Dense(64, activation="relu")(x)
    x = layers.Dense(32, activation="relu")(x)
    x = layers.Flatten()(x)
    x = layers.Dense(16,activation="relu")(x)
    x = layers.Dense(1,activation='sigmoid')(x)


    # x = layers.Dense(64, activation="softplus")(x)
    # x = layers.Dense(32, activation="softplus")(x)
    # x = layers.Dense(16, activation="softplus")(x)
    # x = layers.Dense(4,activation="softplus")(x)
    # x = layers.Dense(1,activation='sigmoid')(x)


    return models.Model(inputs=board1d, outputs=x)





# Establish the model's topography.
model = create_model()
model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.01), loss='mean_squared_error')
model.summary()
history = model.fit(inputs2, outputs,
          batch_size=2048,
          epochs=200,
          verbose=1,
          validation_split=0.15,
          callbacks=[callbacks.ReduceLROnPlateau(monitor='loss', patience=10),
                     callbacks.EarlyStopping(monitor='loss', patience=15, min_delta=1e-4)])

# epochs = history.epoch
# hist = pd.DataFrame(history.history)
# mse = hist["mean_squared_error"]
# plot_the_loss_curve(epochs, mse)

# model.save("chessmodel")



myarr=fenTolist2("rnbqkb1r/pppppppp/5n2/7Q/4P3/8/PPPP1PPP/RNB1KBNR b KQkq - 2 2") #-1262
myarr15=fenTolist2("3r2k1/4b1pp/p1n2N2/1p2pP2/P5R1/8/1PP2PPP/6K1 b - - 0 24") #-818
myarr2=fenTolist2("rnbqkb1r/pppppppp/B4n2/8/4P3/8/PPPP1PPP/RNBQK1NR b KQkq - 2 2") # -400
myarr3=fenTolist2("rnbqkb1r/pppppppp/8/3nP3/8/8/PPPPKPPP/RNBQ1BNR b kq - 2 3") #-230
myarr4=fenTolist2("rnbqkb1r/pppppppp/8/4P3/4n3/8/PPPP1PPP/RNBQKBNR w KQkq - 1 3")  #184
myarr5=fenTolist2("rnbqkb1r/pppppp1p/5n2/4P1p1/8/8/PPPP1PPP/RNBQKBNR w KQkq - 0 3") #412
myarr6=fenTolist2("rn1qkb1r/ppp1pppp/1n1P4/8/2PP4/7b/PP3PPP/RNBQKBNR w KQkq - 1 6")  #625
myarr7=fenTolist2("r1b1k2r/pp1nppbp/2np2p1/q7/2PP4/2B2N2/1P3PPP/RN1QKB1R w KQkq - 0 11")  #1056
myarr=np.expand_dims(myarr,0)
myarr15=np.expand_dims(myarr15,0)
myarr2=np.expand_dims(myarr2,0)
myarr3=np.expand_dims(myarr3,0)
myarr4=np.expand_dims(myarr4,0)
myarr5=np.expand_dims(myarr5,0)
myarr6=np.expand_dims(myarr6,0)
myarr7=np.expand_dims(myarr7,0)
print(model.predict(myarr)[0][0])
print(model.predict(myarr15)[0][0])
print(model.predict(myarr2)[0][0])
print(model.predict(myarr3)[0][0])
print(model.predict(myarr4)[0][0])
print(model.predict(myarr5)[0][0])
print(model.predict(myarr6)[0][0])
print(model.predict(myarr7)[0][0])

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