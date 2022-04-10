import chess

numbers=["1","2","3","4","5","6","7","8"]
fen_eval={
  'p':-10,  'P':10,
  'n': -40,'N': 40,
  'r': -80,'R': 80,
  'b': -40,'B': 40,
  'q': -160,'Q': 160,
  'k': -1000,'K':  1000,
  '/':  0,'1':  0,
  '2':  0,
  '3':  0,
  '4':  0,
  '5':  0,
  '6':  0,
  '7':  0,
  '8':  0,
}



evals_P=[ ###taşların konuma göre puanlandırılması
[12,12,12,12,12,12,12,12],
[6,6,6,6,6,6,6,6],
[5,5,5,5,5,5,5,5],
[4,4,4,4,4,4,4,4],
[3,3,3,3,3,3,3,3],
[2,2,2,2,2,2,2,2],
[1,1,1,1,1,1,1,1],
[0,0,0,0,0,0,0,0],
]
evals_N=[
[6,6,6,6,6,6,6,6],
[7,8,8,8,8,8,8,7],
[7,7,10,10,10,10,7,7],
[5,5,10,10,10,10,5,5],
[3,3,3,3,3,3,3,3],
[1,1,1,1,1,1,1,1],
[1,1,1,1,1,1,1,1],
[0,0,0,0,0,0,0,0],
]
evals_B=[
[1,1,0,0,0,0,1,1],
[2,2,1,1,1,1,2,2],
[3,4,4,6,6,4,4,3],
[4,6,8,8,8,8,6,4],
[4,6,8,8,8,8,6,4],
[3,4,4,6,6,4,4,3],
[2,2,1,1,1,1,2,2],
[1,1,0,0,0,0,1,1],
]
evals_R=[
[4,4,4,4,4,4,4,4],
[3,3,3,3,3,3,3,3],
[2,2,2,2,2,2,2,2],
[1,1,1,1,1,1,1,1],
[0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0],
[0,1,1,1,1,1,0,0],
]
evals_K=[
[0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0],
[1,1,1,2,4,1,1,1],
]
evals_Q=[
[0,0,0,0,0,0,0,0],
[0,6,6,6,6,6,6,0],
[0,6,12,12,12,12,6,0],
[0,6,12,22,22,12,6,0],
[0,6,12,22,22,12,6,0],
[0,6,12,22,22,12,6,0],
[0,6,6,6,6,6,6,0],
[0,0,0,0,0,0,0,0],
]
def isNumber(s):
  return numbers.__contains__(s)

def pointEvaluator(fenString): ###taşları puanlarına göre toplayıp değerlendirme yapar
  score=0
  for index,char in enumerate(fenString):
    if char== " ":
      break
    score = score + fen_eval[char]
  return score


def Evaluate(fenString,player):
  return (pointEvaluator(fenString) + 2* positionEvaluator(fenString, player)+threadEvaluator(fenString));

def threadEvaluator(fenString):
  board = chess.Board(fenString)
  value=0
  for index in range(0, 64):
    if (board.is_attacked_by(True, index)):
      value=value+3
  for index in range(0, 64):
    if (board.is_attacked_by(False, index)):
      value = value - 3
  return value

def positionEvaluator(fenString,player):###taşları posizyonlarına göre numaralandırır
  score = 0                             ## true-> white
  columnindex=0
  rowindex=0
  for index, char in enumerate(fenString):
    if char == " ":
      break
    elif char == "/":
      rowindex=rowindex+1
      columnindex=0
    elif isNumber(char):
      numval=int(char)
      columnindex=columnindex+numval
    else:
      if player:
        if char == "P":
          score=score+evals_P[rowindex][columnindex]
        if char == "R":
          score=score+evals_R[rowindex][columnindex]
        if char == "N":
          score=score+evals_N[rowindex][columnindex]
        if char == "B":
          score=score+evals_B[rowindex][columnindex]
        if char == "Q":
          score=score+evals_Q[rowindex][columnindex]
        if char == "K":
          score=score+evals_K[rowindex][columnindex]
      else:
        if char == "p":
          score = score - evals_P[7-rowindex][7-columnindex]
        if char == "r":
          score = score - evals_R[7-rowindex][7-columnindex]
        if char == "n":
          score = score - evals_N[7-rowindex][7-columnindex]
        if char == "b":
          score = score - evals_B[7-rowindex][7-columnindex]
        if char == "q":
          score = score - evals_Q[7-rowindex][7-columnindex]
        if char == "k":
          score = score - evals_K[7-rowindex][7-columnindex]

  return 2*score







