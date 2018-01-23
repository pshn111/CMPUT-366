#!/usr/bin/env python

"""
  Author: Adam White, Matthew Schlegel, Mohammad M. Ajallooeian, Andrew
  Jacobsen, Victor Silva, Sina Ghiassian
  Purpose: Implementation of the interaction between the Gambler's problem environment
  and the Monte Carlon agent using RL_glue. 
  For use in the Reinforcement Learning course, Fall 2017, University of Alberta

"""

from rl_glue import *  # Required for RL-Glue
RLGlue("maze_env", "DQ_agent")

from DQ_agent import setN, setStep_size
import numpy as np
import matplotlib.pyplot as plt
from utils import rand_in_range
import pickle


	    
if __name__ == "__main__":
	n=5
	alist = [0.03125,0.0625,0.125,0.25,0.5,1.0]
	data = np.zeros((6))
	
	
	for run in range(10):
		b=rand_in_range(50)	
		np.random.seed(b)
		
		for i in range(6):
			setN(n)
			setStep_size(alist[i])
	    
			RL_init()
			
			episodes = 0
			while episodes<50:
				RL_episode(2000)
				steps = RL_num_steps()
				data[i] += steps
				episodes +=1
			RL_cleanup()
	    	    
	
	y = data/(10*50)
	

	plt.show()
	
	plt.plot(alist,y,label = "n=5")
	plt.xticks([0,0.03125,0.0625,0.125,0.25,0.5,1.0])
	plt.ylim(1,100)
	plt.yticks([0,20,40,60,80,100])
	plt.ylabel('steps per episode')
	plt.xlabel('alpha')
	plt.legend()
	plt.show()
		

	
