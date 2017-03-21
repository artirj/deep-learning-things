import numpy as np
from pprint import pprint as pp
from collections import defaultdict
import pickle
def blank_state(flat=False):
	"""Generate a 9x1 zero vector as blank state"""
	if flat:
		return State(np.zeros((9,),dtype=np.int32))
	return State(np.zeros((3, 3),dtype=np.int32))

def softmax(array):
	return np.exp(array-np.max(array))/(np.exp(array-np.max(array)).sum())

def run_trial(player1,n=5):
	board=Board()
	player=Player(board,1)
	player.Q=player1.Q
	player.epsilon=0	
	player2=Player(board,2,beh='random')
	#player2=Player(board,2,beh='fixed')
	win_1=0
	for episode in range(n):
		board.clear()
		for i in range(6):
			action=player.move()		
			board.update(action,1)	
			if(board.get_condition()==1): win_1+=1
			if(board.get_condition()==2): win_1-=1
			if(board.get_condition()<4): break	
			action=player2.move()
			board.update(action,2)		
			if(board.get_condition()==1): win_1+=1
			if(board.get_condition()==2): win_1-=1
			if(board.get_condition()<4): break
	return win_1/n

class Board():
		def __init__(self):
			self.state=State(blank_state())


		def show(self):
			"""Displays board"""
			pp(self.state.astype(np.int64))


		def start_game(self,active_player=1):
			self.active_player=1

		def clear(self):
			self.state=State(blank_state())


		def set_state(self,state):
			
			if(type(state)!=np.array):
				state=State(np.array(state))
			if(state.shape==(3,3)):
					self.state=State(state)
			elif(state.shape==(9,)):
					self.state=State(state.reshape(3,3))
			else:
				raise Exception("Wrong state format")			



		def get_state(self,flat=False):
			"""Get the state of the board
			Returns:
			Numpy array, 9x1"""
			if flat:
				return self.state.reshape(9,)
			return self.state


		def get_empty(self):
			"""Get empty squares aka allowed moves
			Returns:
			Numpy array (variable size)"""
			return np.argwhere(self.state.to_9()==0).flatten()


		def available_states(self,player):
			"""
			Returns:
			array of possible states
			"""
			b=self.get_empty()
			states=np.array(np.broadcast_to(self.state.to_9(),(len(b),9,)))
			for i,pos in enumerate(b):
		    		states[i][pos]=player
			return State(states.reshape(len(b),3,3))



		def get_condition(self,input_state=''):
			"""Returns condition of the board
			1: Player 1 wins
			2: Player 2 wins			
			3: Game ended without winner
			4: Game still on"""
			if(type(input_state)==str):
				state=self.state
			else:
				state=input_state

			if((state.diagonal()==1).all() or
					(state[::-1].diagonal()==1).all() or
					((state==1).sum(0)==3).any() or
					((state==1).sum(1)==3).any()):
					return 1

			elif((state.diagonal()==2).all() or
					(state[::-1].diagonal()==2).all() or
					((state==2).sum(0)==3).any() or
					((state==2).sum(1)==3).any()):
					return 2
			elif(len(self.get_empty())==0):
					return 3
			else:
				return 4


		def update(self,action,player):
			if(type(action)!=None):
				self.state.to_9()[action]=player

class State(np.ndarray):

	def __new__(cls,inputarr):
		return np.asarray(inputarr).view(cls)


	def to_9(self):
		return self.reshape(9,)


	def to_tuple(self):
		return tuple(self.to_9())

class Player():

	def __init__(self,board,n,beh='qlearn',alpha=0.6,gamma=0.9999):
		self.n=n
		self.Q=defaultdict(np.float32)
		self.board=board
		self.alpha=alpha
		self.gamma=gamma
		if(beh=='random'):
			self.epsilon=1
		if(beh=='fixed'):
			self.epsilon=0.5
		else:
			self.epsilon=0.9
		self.beh=beh
		if(n==1):
			self.other=2
		else:
			self.other=1

	def update_epsilon(self,episode):
		self.epsilon=0.9*np.exp(-1e-5*episode)


	def set_Q(self,action,val):
		self.Q[(self.board.get_state().to_tuple(),action)]=val


	def update_Q(self,action):
		"""Updates the Q-value of a given state"""
		if(type(action)!=None):			
			self.Q[(self.board.get_state().to_tuple(),action)]+=self.alpha*(self.reward(action)
				+self.gamma*self.max_q(action)-self.get_Q(action))


	def max_q(self,action):
		"""Returns the maximum of the q-values of the states accessible from the next state"""
		return np.argmax(self.eval_board(action))

	def eval_board(self,action=-1):
		"""Returns the Q-values for the current allowed actions"""
		actions=self.board.get_empty()
		if(action!=-1):	
			pseudostate=State(self.board.get_state())
			pseudostate.to_9()[action]=self.n
			val=[self.get_Q(actions[i],pseudostate) for i in range(actions.size)]
			if(len(val)==0):
				val=self.reward(action,state='True')
			return val
		else:
			return_values=np.zeros((9,))
			return_values[actions]=[self.get_Q(actions[i]) for i in range(actions.size)]
			return return_values

	def allowed_q(self):
		"""Returns the q-values of the allowed actions"""
		actions=self.board.get_empty()
		return_values=[self.get_Q(actions[i]) for i in range(actions.size)]
		return return_values,actions



	def move(self,move_type='softmax'):
		"""Makes a move using softmax"""
		dice=np.random.uniform()
		if(dice<self.epsilon):
			move=np.random.permutation(self.board.get_empty())[:1]
			if(move.size==0):
				return None
		else:
			qvalues,allowed=self.allowed_q()
			if(move_type=='max'):
				move=allowed[np.argmax(qvalues)]
			elif(move_type=='softmax'):
				move=np.random.choice(allowed,p=softmax(qvalues))
			else:
				raise Exception("Invalid move_type! Available types are softmax and max")
		
		return np.asscalar(move)


	def get_Q(self,action,state=0):		
		if(type(state)==int):
			state=self.board.get_state()
		if(type(state)==(np.ndarray or list) ):
			state=State(state)
		

		return self.Q[(State(state).to_tuple(),action)]


	def reward(self,action,state='False'):
		"""This returns the reward if an action is taken from the current state"""
		if(state=='False'):
			pseudostate=State(self.board.get_state())
			pseudostate.to_9()[action]=self.n
			n=self.board.get_condition(pseudostate)
		else:
			n=self.board.get_condition()

		if(n==4):
			R=0
		elif(n==self.n):
			R=1000
		elif(n==self.other):
			R=-1000
		elif(n==3):
			R=-1
		
		return R

	def save_Q(self):
		pickle.dump(self.Q,open("playerQ.p","wb"))
	def load_Q(self):
		self.Q=pickle.load(open("playerQ.p",'rb'))









