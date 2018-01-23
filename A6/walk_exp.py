#!/usr/bin/env python

"""
  Author: Adam White, Matthew Schlegel, Mohammad M. Ajallooeian, Andrew
  Jacobsen, Victor Silva, Sina Ghiassian
  Purpose: Implementation of the interaction between the Gambler's problem environment
  and the Monte Carlon agent using RL_glue. 
  For use in the Reinforcement Learning course, Fall 2017, University of Alberta

"""

from rl_glue import *  # Required for RL-Glue


from part2_agent1 import getReturn
from part2_agent2 import getReturn1
from part2_agent3 import getReturn2
from rndmwalk_policy_evaluation import compute_value_function
import math
import numpy as np
import matplotlib.pyplot as plt
from utils import rand_in_range
import pickle


	    
if __name__ == "__main__":
	rmse1=np.zeros(2000)
	rmse2=np.zeros(2000)
	rmse3=np.zeros(2000)
	np.random.seed(1)
	v = compute_value_function()[1:]
	
	for run in range(10):
		#print "Run" + str(run)
		
	        for i in range(3):
		
			
			
			if i == 0:
				RLGlue("walk_env", "part2_agent1")
				RL_init()
				for m in range(2000):
					RL_episode(5000)
					valueFunction = getReturn()
					rmse1[m] += math.sqrt( np.sum((v - valueFunction)*(v - valueFunction))/1000.0 )
					
				RL_cleanup()
				
			elif i ==1:
				RLGlue("walk_env", "part2_agent2")
				RL_init()
				for m in range(2000):
					RL_episode(5000)
					valueFunction = getReturn1()
					rmse2[m] += math.sqrt( np.sum((v - valueFunction)*(v - valueFunction))/1000.0 )				
				RL_cleanup()
				
			elif i ==2:
				RLGlue("walk_env", "part2_agent3")
				RL_init()
				for m in range(2000):
					RL_episode(5000)
					valueFunction = getReturn2()
					rmse3[m] += math.sqrt( np.sum((v - valueFunction)*(v - valueFunction))/1000.0 )				
				RL_cleanup()
			
			
	    	    
	

	rmse1=rmse1/10
	rmse2=rmse2/10
	rmse3=rmse3/10
	plt.show()

	plt.plot(rmse1,label = "agent1")
	plt.plot(rmse2,label = "agent2")
	plt.plot(rmse3,label = "agent3")
	plt.xlim([1,2000])
	plt.ylabel('RMSVE')
	plt.xlabel('Episodes')
	plt.legend()
	plt.show()	
