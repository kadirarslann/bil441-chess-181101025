import tkinter
import tkinter as tk
import threading
window = tk.Tk()
validfen='rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'
numbers=["1","2","3","4","5","6","7","8"]


def isNumber(s):
    return  numbers.__contains__(s)

def initLabel(value,row,column):
    label = tk.Label(master=window, text=value)
    label.config(font=("Courier", 44))
    label.grid(row=row, column=column, padx=1, pady=1)
    label.pack

def fentochess(fenstring):
    rowIndex=0;
    columnIndex=0;
    charIndex=0;

    while (fenstring[charIndex] != ' '):
        if(isNumber(fenstring[charIndex])):
            numval=int(fenstring[charIndex])
            while(numval!=0):
                # print(0,end=" ")
                numval=numval-1
                initLabel("0",rowIndex,columnIndex)
                columnIndex=columnIndex+1
            charIndex = charIndex + 1
        elif(fenstring[charIndex]=='/'):
            # print()
            charIndex = charIndex + 1
            rowIndex=rowIndex+1
            columnIndex = 0
        else:
            # print(fenstring[charIndex],end=" ")
            initLabel(fenstring[charIndex],rowIndex,columnIndex)
            charIndex = charIndex + 1
            columnIndex = columnIndex + 1

for i in range(8):
    window.columnconfigure(i, weight=1, minsize=50)
    window.rowconfigure(i, weight=1, minsize=50)

def initVisualizer():
    fentochess(validfen)

# window.mainloop()
