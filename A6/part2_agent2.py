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


iht = IHT(2048)

w = None
x = None
z = None
current_state = None
prev_state = None
actual_state = None
alpha = 0.01/50
gamma = 1.0



def agent_init():
    global w,x,z,current_state,prev_state,actual_state
    """
    Hint: Initialize the variables that need to be reset before each run begins
    Returns: nothing
    """
    
    #print("n = %d" %(n))
    z = np.zeros(1000)
    w = np.zeros(1200)
    current_state = np.zeros(1)
    prev_state = np.zeros(1)
    actual_state = np.zeros(1)
    
    return    

    #initialize the policy array in a smart way

def agent_start(state):
    global w,x,z,current_state,prev_state,actual_state
    """
    Hint: Initialize the variavbles that you want to reset before starting a new episode
    Arguments: state: numpy array
    Returns: action: integer
    """
    # pick the first action, don't forget about exploring starts 
    
    current_state[0] = float(state[0]/200.0)
    prev_state[0] = current_state[0]
    actual_state[0] = state[0]
    
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
    global w,x,z,current_state,prev_state,actual_state
    """
    Arguments: reward: floting point, state: integer
    Returns: action: integer
    """
    # select an action, based on Q
    
    current_state[0] = float(state[0]/200.0)
    
    Cstate = np.zeros(1200)
    Pstate = np.zeros(1200)    
    
    current_x = tiles(iht,50,current_state)
    
    for i in current_x:
	    Cstate[i] = 1
	    
    prev_x =  tiles(iht,50,prev_state)
    
    for j in prev_x:
	    Pstate[j] = 1
		    
    
    w = w + alpha*(reward+gamma*np.dot(w,Cstate) - np.dot(w,Pstate))*Pstate
    
    z[actual_state[0]] = np.dot(w,Pstate)
    
    prev_state[0] = current_state[0]
    actual_state[0] = state[0]
    
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

def agent_end(reward):
    global w,x,z,current_state,prev_state,actual_state
    """
    Arguments: reward: floating point
    Returns: Nothing
    """
    # do learning and update pi
    
    Pstate = np.zeros(1200)

    prev_x =  tiles(iht,50,prev_state)

    for i in prev_x:
	    Pstate[i] = 1


    w = w + alpha*(reward- np.dot(w,Pstate))*Pstate

    z[actual_state[0]] = np.dot(w,Pstate)
    
    return

def agent_cleanup():
    """
    This function is not used
    """
    # clean up
    return

def agent_message(in_message): # returns string, in_message: string
    global w,x,z,current_state,prev_state,actual_state
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

def getReturn1():
    return z
