#!/usr/bin/env python

"""
  Author: Adam White, Matthew Schlegel, Mohammad M. Ajallooeian, Sina Ghiassian
  Purpose: Skeleton code for Monte Carlo Exploring Starts Control Agent
           for use on A3 of Reinforcement learning course University of Alberta Fall 2017
 
"""

from utils import rand_in_range, rand_un
import numpy as np
import random
import pickle

w = None
x = None
current_state = None
prev_state = None
alpha = 0.5
gamma = 1.0




def agent_init():
    global w,x,current_state,prev_state
    """
    Hint: Initialize the variables that need to be reset before each run begins
    Returns: nothing
    """
    
    #print("n = %d" %(n))
    
    x = np.identity(1000)
    w = np.zeros(1000)
    current_state = np.zeros(1)
    prev_state = np.zeros(1)
    
    
    return    

    #initialize the policy array in a smart way

def agent_start(state):
    global w,x,current_state,prev_state
    """
    Hint: Initialize the variavbles that you want to reset before starting a new episode
    Arguments: state: numpy array
    Returns: action: integer
    """
    # pick the first action, don't forget about exploring starts 
    
    current_state[0] = state[0]
    prev_state[0] = current_state[0]
    
    if random.randint(0,1) : #go right
	rnum = random.randint(1,100)
	if rnum+state[0]>1000:
		action = 1000-state[0]
	else:
		action = rnum

    else: #go left
	rnum = (random.randint(1,100))*(-1)
	if rnum+state[0]<1:
		action = state[0]*(-1)
	else:
		action = rnum 
		
    
    
    return action
	
    


def agent_step(reward, state): # returns NumPy array, reward: floating point, this_observation: NumPy array
    global w,x,current_state,prev_state
    """
    Arguments: reward: floting point, state: integer
    Returns: action: integer
    """
    # select an action, based on Q
    
    current_state[0] = state[0]
    w = w + alpha*(reward+gamma*w[current_state[0]-1]- w[prev_state[0]-1])*x[prev_state[0]-1]
    
    if random.randint(0,1) : #go right
	rnum = random.randint(1,100)
	if rnum+state[0]>1000:
		action = 1000-state[0]
	else:
		action = rnum

    else: #go left
	rnum = (random.randint(1,100))*(-1)
	if rnum+state[0]<1:
		action = state[0]*(-1)
	else:
		action = rnum  
		
    prev_state[0] = current_state[0]    
	    
    
    return action

def agent_end(reward):
    global w,x,current_state,prev_state
    """
    Arguments: reward: floating point
    Returns: Nothing
    """
    # do learning and update pi
    
    w = w + alpha*(reward-w[prev_state[0]-1])*x[prev_state[0]-1]

    return

def agent_cleanup():
    """
    This function is not used
    """
    # clean up
    return

def agent_message(in_message): # returns string, in_message: string
    global w,x,current_state,prev_state
    """
    Arguments: in_message: string
    returns: The value function as a string.
    This function is complete. You do not need to add code here.
    """
    # should not need to modify this function. Modify at your own risk
    if (in_message == 'ValueFunction'):
        return 
    else:
        return "I don't know what to return!!"

def getReturn():
    return w