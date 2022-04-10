from __future__ import absolute_import, division, print_function, unicode_literals
import os

from algos import fenTolist,fenToBoard,fenToArrs,fenTolist2,getBoardValueAlternative
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
train_df = pd.read_csv("./samples/random_evals.csv")  #### dosya okuma
train_df = train_df.reindex(np.random.permutation(train_df.index)) # verileri karıştırma


# inputs = np.zeros(shape=(500000, 64))  # en basit girdi
# inputs2d = np.zeros(shape=(50000, 8,8)) ## matrix girdisi
# inputs3d = np.zeros(shape=(50000, 14,8,8)) ##her bir tas turu icin 8x8 array temsili

inputs2 = np.zeros(shape=(200000, 256)) # best
outputs = np.zeros(shape=(200000,))

for index, item in enumerate(inputs2):
    if (index >= 200000):
        break
    # inputs3d[index] = fenToArrs(train_df["FEN"][index])
    # inputs2d[index]=fenToBoard(train_df["FEN"][index])
    # inputs[index] = fenTolist(train_df["FEN"][index])
    inputs2[index] = fenTolist2(train_df["FEN"][index])
    if(train_df["Evaluation"][index][0]=="#"): ### output value calculation for #+- format
        if (train_df["Evaluation"][index][1]=="+"):
            outputs[index] = getBoardValueAlternative(train_df["Evaluation"][index])
            # print("beyaz avantaj", bbb[index])
        else:
            outputs[index] = -1 * getBoardValueAlternative(train_df["Evaluation"][index])
            # print("siyah avantaj", bbb[index])
        # print(train_df["Evaluation"][index])
    else:  ### normalization
        if(int(train_df["Evaluation"][index])>5000):
            outputs[index] = 5000;
        elif(int(train_df["Evaluation"][index])<-5000):
            outputs[index] = -5000
        else:
            outputs[index] = train_df["Evaluation"][index]

outputs=outputs/1000;### normalization
##############################################################################
def plot_the_loss_curve(epochs, mse): ### ploting error
  """Plot a curve of loss vs. epoch."""
  plt.figure()
  plt.xlabel("Epoch")
  plt.ylabel("Mean Squared Error")
  plt.plot(epochs, mse, label="Loss")
  plt.legend()
  plt.ylim([mse.min()*0.99, mse.max()*1.01])
  plt.show()
print("Defined the plot_the_loss_curve function.")
#################################################################################################################

def create_model(): ##model creation
    board1d = layers.Input(shape=(256,))
    x = board1d

    x = layers.Dense(64, activation="softplus")(x)  # best
    x = layers.Dense(16, activation="softplus")(x)
    x = layers.Dense(4, activation="softplus")(x)
    x = layers.Dense(1, activation='sigmoid')(x)

    return models.Model(inputs=board1d, outputs=x)



model = create_model()
model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.01), loss='mean_squared_error',metrics=[tf.keras.metrics.MeanSquaredError()])
model.summary()
history = model.fit(inputs2, outputs,  #training
          batch_size=4096,
          epochs=150,
          verbose=1,
          validation_split=0.15,
          shuffle=True,
          callbacks=[callbacks.ReduceLROnPlateau(monitor='loss', patience=10),
                     callbacks.EarlyStopping(monitor='loss', patience=15, min_delta=1e-3)])

epochs = history.epoch
hist = pd.DataFrame(history.history)
mse = hist["mean_squared_error"]
plot_the_loss_curve(epochs, mse)

# model.save("chess_model") ### save model


##some rondam value testing
myarr1=fenTolist2("rnbqkb1r/pppppppp/5n2/7Q/4P3/8/PPPP1PPP/RNB1KBNR b KQkq - 2 2") #-1262
myarr2=fenTolist2("3r2k1/4b1pp/p1n2N2/1p2pP2/P5R1/8/1PP2PPP/6K1 b - - 0 24") #-818
myarr3=fenTolist2("rnbqkb1r/pppppppp/B4n2/8/4P3/8/PPPP1PPP/RNBQK1NR b KQkq - 2 2") # -400
myarr4=fenTolist2("rnbqkb1r/pppppppp/8/3nP3/8/8/PPPPKPPP/RNBQ1BNR b kq - 2 3") #-230
myarr5=fenTolist2("rnbqkb1r/pppppppp/8/4P3/4n3/8/PPPP1PPP/RNBQKBNR w KQkq - 1 3")  #184
myarr6=fenTolist2("rnbqkb1r/pppppp1p/5n2/4P1p1/8/8/PPPP1PPP/RNBQKBNR w KQkq - 0 3") #412
myarr7=fenTolist2("rn1qkb1r/ppp1pppp/1n1P4/8/2PP4/7b/PP3PPP/RNBQKBNR w KQkq - 1 6")  #625
myarr8=fenTolist2("r1b1k2r/pp1nppbp/2np2p1/q7/2PP4/2B2N2/1P3PPP/RN1QKB1R w KQkq - 0 11")  #1056
myarr1=np.expand_dims(myarr1,0)
myarr2=np.expand_dims(myarr2,0)
myarr3=np.expand_dims(myarr3,0)
myarr4=np.expand_dims(myarr4,0)
myarr5=np.expand_dims(myarr5,0)
myarr6=np.expand_dims(myarr6,0)
myarr7=np.expand_dims(myarr7,0)
myarr8=np.expand_dims(myarr8,0)
print(model.predict(myarr1)[0][0])
print(model.predict(myarr2)[0][0])
print(model.predict(myarr3)[0][0])
print(model.predict(myarr4)[0][0])
print(model.predict(myarr5)[0][0])
print(model.predict(myarr6)[0][0])
print(model.predict(myarr7)[0][0])
print(model.predict(myarr8)[0][0])
