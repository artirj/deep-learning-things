import numpy as np
import sys, os
myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + '/../')
import deeptactoe
from deeptactoe import game as g
def test_blank():
	assert np.array_equal(g.blank_state(),np.zeros((3,3),dtype=np.int32))

def test_board_set_get():
	board=g.Board()
	board.set_state(np.array([1,2,0,0,0,1,2,1,2]).reshape((3,3)))
	assert np.array_equal(board.get_state(),np.array([1,2,0,0,0,1,2,1,2]).reshape((3,3)))
	empty=board.get_empty()
	assert np.array_equal(empty,np.array([2,3,4]))
def test_board_empty():
	board=g.Board()
	board.set_state(np.array([0,0,0,0,0,0,0,0,0]).reshape((3,3)))
	empty=board.get_empty()
	assert np.array_equal(empty,np.arange(9))

def test_board_cond():
	board=g.Board()

	board.set_state(np.array([1,1,1,0,0,0,0,0,0]).reshape((3,3)))
	cond=board.get_condition()
	assert cond==1

	board.set_state(np.array([2,0,0,0,2,0,0,0,2]).reshape((3,3)))
	cond=board.get_condition()
	assert cond==2

	board.set_state(np.array([2,1,0,1,0,1,2,0,0]).reshape((3,3)))
	cond=board.get_condition()
	assert cond==4

	board.set_state(np.array([1,2,1,2,2,1,1,1,2]).reshape((3,3)))
	cond=board.get_condition()
	assert cond==3

	board.set_state(np.array([2,2,1,2,2,1,1,1,2]).reshape((3,3)))
	cond=board.get_condition()
	assert cond==2

def test_board_available_states():
	board=g.Board()
	board.set_state([2,1,0,1,2,1,2,0,2])
	assert np.array_equal(board.available_states(1),np.array([np.array([2,1,1,1,2,1,2,0,2]).reshape(3,3),np.array([2,1,0,1,2,1,2,1,2]).reshape(3,3)]))


def test_board_update():
	board=g.Board()
	board.update(4,1)
	assert board.get_state()[1,1]==1

def test_player_Q():
	board=g.Board()
	player=g.Player(board,1)
	player.set_Q(1,56)
	assert player.get_Q(1)==56


def test_player_rewards():
	board=g.Board()
	player=g.Player(board,1)
	assert player.reward(3)==0
	board.set_state([1,1,1,0,0,0,0,0,0])
	assert player.reward(3)==100
	board.set_state([2,2,2,0,0,0,0,0,0])
	assert player.reward(3)==-100

def test_player_move():
	board=g.Board()
	player=g.Player(board,1)
	player.epsilon=1
	board.set_state([2,1,2,1,2,1,2,0,2])
	assert player.move()==7
	player=g.Player(board,2)
	player.epsilon=1
	board.set_state([2,1,2,1,2,1,2,0,2])
	assert player.move()==7

def test_player_bestmove():
	board=g.Board()
	player=g.Player(board,1)
	board.set_state([2,1,2,1,2,1,2,0,2])
	player.set_Q(7,10)
	assert np.array_equal(player.eval_board(),[  0.,   0.,   0.,   0.,   0.,   0.,   0.,  10.,   0.])
	board.set_state([0,1,2,1,2,1,2,0,2])
	player.set_Q(0,20)
	player.set_Q(7,10)
	assert np.array_equal(player.eval_board(),[ 20.,   0.,   0.,   0.,   0.,   0.,   0.,  10.,   0.])

def test_player_move():
	board=g.Board()
	player=g.Player(board,1)
	board.set_state([1,0,1,0,2,0,0,0,0])
	player.set_Q(1,10)
	player.epsilon=0
	assert player.move()==1


	

def test_state():
	s=g.State(np.array([2,1,0,1,2,1,2,0,2]))
	assert np.array_equal(s,np.array([2,1,0,1,2,1,2,0,2]))
	assert s.to_9().shape==(9,)
	assert type(s.to_tuple())==tuple


def test_softmax():
	assert np.array_equal(g.softmax([1,1]),np.array([0.5,0.5]))
	assert np.array_equal(g.softmax([1,1,1]),np.array([1/3,1/3,1/3]))


