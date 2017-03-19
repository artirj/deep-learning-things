import numpy as np
from deeptactoe.game import blank_slate
def test_blank():
	assert blank_slate()==np.array[0,0,0,0,0,0,0,0,0]
