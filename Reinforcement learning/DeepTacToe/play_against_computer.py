import deeptactoe
from deeptactoe import game as g
from pprint import pprint as pp
import numpy as np
board=g.Board()
player1=g.Player(board,1)
player2=g.Player(board,2,beh='human')
player1.epsilon=1
try:
	player1.load_Q()
except:
	print("Not possible to load Q matrix. Computer will play at random")
def sanitise(board,move):
	if(np.any(np.equal(board.get_empty(),move))):
		return False
	else:
		print("Illegal move!")
		return True
def resolution(board):
	case=board.get_condition()
	if(case==1):
		print("The Computer wins")
	elif(case==2):
		print("Human wins")
	else:
		print("Tie")

board.start_game()
print('Player 1 (CPU) begins because that\'s how the code works\n')
flag=1
while flag:
	okmove=True
	print('Player 1 (CPU) moves\n')
	action=player1.move()
	board.update(action,1)
	board.show()
	if(board.get_condition()<4):
		resolution(board)
		break
	print('Player 2 (Human) moves\n')
	while(okmove):
		while 1:
			try:
				human_move = input('Input your move: (0-9) ')
				break
			except:
				print("Invalid move")
			
		okmove=sanitise(board,human_move)
		print(okmove)
		if(~okmove):
			board.update(human_move,2)
			board.show()
		if(board.get_condition()<4):
			resolution(board)
			flag=0