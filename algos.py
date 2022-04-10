
import numpy as np
import pandas as pd
import chess
numbers=["1","2","3","4","5","6","7","8"]

def isNumber(s):
    return  numbers.__contains__(s)
def getNum(s):
    if s == "P":return 1
    if s == "R":return 5
    if s == "N":return 3
    if s == "B":return 3
    if s == "Q":return 9
    if s == "K":return 16
    if s == "p":return -1
    if s == "r":return -5
    if s == "n":return -3
    if s == "b":return -3
    if s == "q":return -9
    if s == "k":return -16

def getBoardValueAlternative(value):
    if len(value)==4:
        return 2000+ (15-(10*int(value[2])+int(value[3])))*200
    else:
        return 2000+(15 - (int(value[2]))) * 200

def getArrIndex(s):
    if s == "P":return 0
    if s == "R":return 1
    if s == "N":return 2
    if s == "B":return 3
    if s == "Q":return 4
    if s == "K":return 5
    if s == "p":return 6
    if s == "r":return 7
    if s == "n":return 8
    if s == "b":return 9
    if s == "q":return 10
    if s == "k":return 11
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

def fenTolist2(fenString):
    a = np.zeros(shape=(256))
    board = chess.Board(fenString)
    columnindex = 0
    rowindex = 0
    for index, char in enumerate(fenString):
        if char == " ":
            break
        elif char == "/":
            pass
        elif isNumber(char):
            numval = int(char)
            while (numval != 0):
                # a[columnindex] = 0
                columnindex = columnindex + 1
                numval = numval - 1
        else:
            if(getNum(char)>0):
                a[columnindex] = getNum(char)
            else:
                a[64+columnindex] = getNum(char)
            columnindex = columnindex + 1

    for index in range(0,64):
        if(board.is_attacked_by(True, index)):
            a[128+index]=1
    for index in range(0, 64):
        if (board.is_attacked_by(False, index)):
            a[192+index]=1
    return a




def fenToBoard(fenString):
    a = np.zeros(shape=(8, 8))
    # print(a)
    columnindex = 0
    rowindex = 0
    for index, char in enumerate(fenString):
        if char == " ":
            break
        elif char == "/":
            rowindex = rowindex + 1
            columnindex = 0
        elif isNumber(char):

            numval = int(char)
            while(numval!=0):
                a[rowindex][columnindex]=0
                columnindex = columnindex + 1
                numval=numval-1
        else:
            # print(rowindex, "  ", columnindex)
            a[rowindex][columnindex]=getNum(char)
            columnindex = columnindex + 1
    return a

def fenToArrs(fenString):
    a = np.zeros(shape=(14,8,8))
    board = chess.Board(fenString)
    columnindex = 0
    rowindex = 0
    for index, char in enumerate(fenString):
        if char == " ":
            break
        elif char == "/":
            rowindex = rowindex + 1
            columnindex = 0
        elif isNumber(char):
            numval = int(char)
            while (numval != 0):
                columnindex = columnindex + 1
                numval = numval - 1
        else:
            # print(rowindex, "  ", columnindex)
            if char == 'P':
                a[0][rowindex][columnindex] = 1
            if char == 'R':
                a[1][rowindex][columnindex] = 1
            if char == 'B':
                a[2][rowindex][columnindex] = 1
            if char == 'N':
                a[3][rowindex][columnindex] = 1
            if char == 'Q':
                a[4][rowindex][columnindex] = 1
            if char == 'K':
                a[5][rowindex][columnindex] = 1
            if char == 'p':
                a[6][rowindex][columnindex] = 1
            if char == 'r':
                a[7][rowindex][columnindex] = 1
            if char == 'b':
                a[8][rowindex][columnindex] = 1
            if char == 'n':
                a[9][rowindex][columnindex] = 1
            if char == 'q':
                a[10][rowindex][columnindex] = 1
            if char == 'k':
                a[11][rowindex][columnindex] = 1
            columnindex = columnindex + 1
    for index in range(0,64):
        if(board.is_attacked_by(True, index)):
            a[12][int(index/8)][index%8]=1
    for index in range(0, 64):
        if (board.is_attacked_by(False, index)):
            a[13][int(index/8)][index%8]=1
    return a

