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

from tiles3 import tiles, IHT


iht = IHT(4096)

w = None
z = None
r = None

current_state = None
prev_state = None
prev_action = None

alpha = 0.1/(8)
gamma = 1.0
tilingNum = 8
size_tilings = 8
lambda_value = 0.9
epsilon = 0




def agent_init():
    global w,z,r,current_state,prev_state,prev_action
    """
    Hint: Initialize the variables that need to be reset before each run begins
    Returns: nothing
    """
    
    #print("n = %d" %(n))
    z = np.zeros(9*9*8*3)
    w = np.full(9*9*8*3,np.random.uniform(-0.001,0))
    current_state = np.zeros(2)
    prev_state = np.zeros(2)
    prev_action = 0
    
    return    

    #initialize the policy array in a smart way

def agent_start(state):
    global w,z,r,current_state,prev_state,prev_action
    """
    Hint: Initialize the variavbles that you want to reset before starting a new episode
    Arguments: state: numpy array
    Returns: action: integer
    """
    # pick the first action, don't forget about exploring starts 
    
    current_state[0] = 8*state[0]/(0.5+1.2) #get current location
    current_state[1] = 8*state[1]/(0.07+0.07)
    	
    action = pickAction(current_state) 
    prev_action = action		
    prev_state = current_state
    
    return action
	
    


def agent_step(reward, state): # returns NumPy array, reward: floating point, this_observation: NumPy array
    global w,z,r,current_state,prev_state,prev_action
    """
    Arguments: reward: floting point, state: integer
    Returns: action: integer
    """
    # select an action, based on Q
    
    td_error = reward
    
    for pointer in tiles(iht,8,prev_state,[prev_action]) : #update last state
	    td_error = td_error - w[pointer] # td error
	    z[pointer] = 1    
    
    current_state[0] = 8*state[0]/(0.5+1.2)
    current_state[1] = 8*state[1]/(0.07+0.07)  
	    
    action = pickAction(current_state)
    
    for pointer in  tiles(iht,8,current_state,[action]):  #get current state location
	    td_error = td_error + gamma*w[pointer]
		    
    w = w + alpha*z*td_error
    z = z * gamma*lambda_value
    prev_action = action
    prev_state = current_state    
    
    
    return action

def agent_end(reward):
    global w,z,r,current_state,prev_state,prev_action
    """
    Arguments: reward: floating point
    Returns: Nothing
    """
    # do learning and update pi
    
    td_error = reward
    
    for pointer in tiles(iht,8,prev_state,[prev_action]) : #update last state
	    td_error = td_error - w[pointer] # td error
	    z[pointer] = 1
	
    w = w + alpha*td_error*z

    
    return

def agent_cleanup():
    """
    This function is not used
    """
    # clean up
    return

def agent_message(in_message): # returns string, in_message: string
    global w,z,r,current_state,prev_state,prev_action
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
    global w,r
    mp = 1.7/50
    mv = 0.14/50
    r = np.zeros((50,50))
    for i in range(50):
	    for j in range(50):
		    q = np.zeros(3)
		    state = [8*(-1.2+mp*i)/(0.5+1.2),8*(-0.07+mv*j)/(0.07+0.07)]
		    for k in range(3):
			    index = np.zeros(9*9*8*3)
			    index[tiles(iht,8,state,[k])] = 1 
			    q[k] = np.dot(w,index)
    
		    r[i][j] = (-1)*np.nanmax(q)
    
        
    return r


def pickAction(state):
    # epsilon-greedy
    if random.random() < epsilon:
	return random.randint(0,2)
    
    y = np.zeros(3)
    
    list1 = tiles(iht,8,state,[0]) 
    list2 = tiles(iht,8,state,[1]) 
    list3 = tiles(iht,8,state,[2]) 
    
    pointer1 = np.zeros(9*9*8*3)
    pointer2 = np.zeros(9*9*8*3)
    pointer3 = np.zeros(9*9*8*3)
    
    pointer1[list1] = 1    
    pointer2[list2] = 1    
    pointer3[list3] = 1
    
    y[0] = np.dot(w,pointer1)
    y[1] = np.dot(w,pointer2)
    y[2] = np.dot(w,pointer3)
    
    action_pointer = np.nanargmax(y)
    return action_pointer    