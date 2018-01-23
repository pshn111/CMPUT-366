#!/usr/bin/env python

"""
  Author: Adam White, Matthew Schlegel, Mohammad M. Ajallooeian, Andrew
  Jacobsen, Victor Silva, Sina Ghiassian
  Purpose: Implementation of the interaction between the Gambler's problem environment
  and the Monte Carlon agent using RL_glue. 
  For use in the Reinforcement Learning course, Fall 2017, University of Alberta

"""

from rl_glue import *  # Required for RL-Glue
RLGlue("windy_env", "sarsa_agent")

import matplotlib.pyplot as plt
import numpy as np
import pickle

if __name__ == "__main__":
    num_episodes = 0
    max_steps = 8000

    counter = 0
    
    x_list=[]
    y_list=[]
    
    RL_init()
    while counter<max_steps:
	    RL_episode(10000)
	    counter += RL_num_steps()
	    num_episodes = RL_num_episodes()
	    x_list.append(counter)
	    y_list.append(num_episodes)
	    
    RL_cleanup()
    
    
    #draw plot
    
    
    plt.show()
    plt.plot(x_list,y_list)
    plt.xlim([0,8000])
    plt.xticks([0,1000,2000,3000,4000,5000,6000,7000,8000])
    plt.xlabel('Time steps')
    plt.ylabel('Episodes')
    plt.legend()
    plt.show()    

    
