import deeptactoe
from deeptactoe import game as g
from pprint import pprint as pp
import matplotlib
matplotlib.use('GTK3Cairo')
import matplotlib.pyplot as plt
import numpy as np
board=g.Board()

player1=g.Player(board,1)
player2=g.Player(board,2,beh='fixed')
player1.epsilon=1

board.start_game()
wins=[]
n_ep=10000

for episode in range(n_ep):
	board.clear()
	for i in range(6):		

		action=player1.move()		
		player1.update_Q(action)
		#player1.update_epsilon(episode)
		board.update(action,1)	
		if(board.get_condition()<4): break
		action=player2.move()
		
		board.update(action,2)		
		if(board.get_condition()<4): break
	if(episode%100==0):
		wins.append(g.run_trial(player1,1))


plt.plot(wins)
plt.show()






