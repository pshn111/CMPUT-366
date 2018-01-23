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

from DQ_agent import setN
import numpy as np
import matplotlib.pyplot as plt
from utils import rand_in_range
import pickle


	    
if __name__ == "__main__":
	nlist = [0,5,50]
	data = np.zeros((3,50))
	
	for run in range(10):
		b=rand_in_range(50)
		np.random.seed(b)
		
	        for i in range(3):
		        setN(nlist[i])
		
			RL_init()
			
			episodes = 0
			while episodes<50:
				RL_episode(2000)
				steps = RL_num_steps()
				data[i][episodes] += steps
				episodes +=1
			RL_cleanup()
	    	    
	
	y0 = data[0]/10
	y5 = data[1]/10
	y50 = data[2]/10

	plt.show()

	plt.plot(y0,label = "n=0")
	plt.plot(y5,label = "n=5")
	plt.plot(y50,label = "n=50")
	plt.xlim([1,50])
	plt.xticks([0,10,20,30,40,50])
	plt.ylim(1,800)
	plt.yticks([15,200,400,600,800])
	plt.ylabel('steps per episode')
	plt.xlabel('episode')
	plt.legend()
	plt.show()	
