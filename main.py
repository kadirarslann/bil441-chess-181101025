import random
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import chess
import evaluators
import time
import numpy as np
from algos import fenTolist2
from chessvisualizer import *
import time
import tensorflow as tf

# validfen = 'rnbqkbnr/pp1ppppp/8/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2'
initialfen="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1" # ilk satranc fen girdisi
validfen1="r1b1r1k1/pp3npp/8/3pn2Q/7q/P2B4/1BPN1PPP/R4RK1 w - - 4 18" ##beyaz az miktar avantajlı
validfen2="3r1rk1/pp2bp1p/2n4B/8/2B1n3/2N2Q2/PPP3P1/2K4R w - - 1 18" ##beyaz fazla miktar avantalı
depths = {} #oynanacak hamleyi hafızada tutar
model = tf.keras.models.load_model("chess_model") ### olusturulmus derin ogrenme modelini import edip kullanma
aiturn = True # deeplearning minmax vs standart minmax

def miniMax(board,depth,player,alfa,beta): ## false =>>>> black
  if(depth == 0):
    global aiturn

    # evals = random.uniform(0.95, 1.05) *evaluators.Evaluate(board.fen(),player) # 2 oyuncuda derin ogrenme modeli minmax kullanır
    # myarr = fenTolist2(board.fen())
    # myarr = np.expand_dims(myarr, 0)
    # evalsai=model.predict(myarr)[0][0]
    # return ( evalsai +(evals/1000))


    # evals = random.uniform(0.95, 1.05) * evaluators.Evaluate(board.fen(), player) # 2 oyuncu standart minmax kulanır
    # return evals

    if aiturn == True:    ### beyaz oyuncu derin ogrenmeli minmax, siyah oyuncu standart minmax kullanır
        evals = random.uniform(0.95, 1.05) * evaluators.Evaluate(board.fen(), player)
        myarr = fenTolist2(board.fen())
        myarr = np.expand_dims(myarr, 0)
        evalsai=model.predict(myarr)[0][0]
        return ( evalsai +(evals/1000))
    else:
        evals = random.uniform(0.95, 1.05) * evaluators.Evaluate(board.fen(), player)
        return evals

  if player:
    maxEval=-100000000
    legalmoves=list(board.legal_moves)
    for index,move in enumerate(legalmoves):
        board.push_san(str(legalmoves[index]))
        eval=miniMax(board,depth-1,False,alfa,beta)
        if(maxEval<=eval):
            maxEval=eval
            depths[(str(depth))]=move
        if(alfa<=eval):
           alfa=eval
        if(beta <= alfa):
           #print("prunnig yapıldı")
           board.pop()
           break
        board.pop()
    return maxEval
  else:
      minEval = 100000000
      legalmoves = list(board.legal_moves)
      for index, move in enumerate(legalmoves):
          board.push_san(str(legalmoves[index]))
          eval = miniMax(board, depth - 1, True,alfa,beta)
          if (minEval >= eval):
              minEval = eval
              depths[(str(depth))] = move
          if (beta >= eval):
             beta = eval
          if (beta <= alfa):
             #print("prunnig yapıldı")
             board.pop()
             break
          board.pop()
      return  minEval

# print(board.legal_moves)
# print(board.fen())
# evaluators.fen_eval
# print(evaluators.fen_eval['k'])
# Press the green button in the gutter to run the script.

board = chess.Board(validfen1)
# board = chess.Board()
minimaxDepth=2;
counter=0
current_time = time.time()
while(1==1):

    # fentochess(board.fen()) ##chess visulizer (makes program very slow)
    # window.update()

    print("----------------------")
    print(board)

    miniMax(board, minimaxDepth, board.turn,-9999999999,+9999999999)
    board.push_san(str(depths[str(minimaxDepth)]))
    if(board.is_checkmate()):
        break
    if (board.is_insufficient_material()):
        print("beraberlik")
        break;
    if(board.is_stalemate()):
        print("beraberlik")
        break;

    aiturn=not aiturn
    counter=counter+1

print("----------------------")
print(board)
# print("----------------------")
# print(board.is_check())
if(board.is_check() and board.turn):
    print("black win","with ",counter," moves")
elif(board.is_check() and (not board.turn)):
    print("white win","with ",counter," moves")
else:
    print("beraberlik", "with ", counter, " moves")
print("game took about:", time.time()-current_time)

# print(board.move_stack[len(board.move_stack)-1]) ### oynanmıs hamleleri listeler
# print(board.ply()) kaç tane yarım hamle yapıldığını sayar

