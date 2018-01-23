#!/usr/bin/env python

"""
  Author: Adam White, Matthew Schlegel, Mohammad M. Ajallooeian, Sina Ghiassian
  Purpose: Skeleton code for Monte Carlo Exploring Starts Control Agent
           for use on A3 of Reinforcement learning course University of Alberta Fall 2017
 
"""

from utils import rand_in_range, rand_un
import numpy as np
import pickle

n = None
Q = None
model = None
actions = {0:[1,0],1:[0,-1],2:[-1,0],3:[0,1]}
prev_action = None 
prev_x = None 
prev_y = None 
#you may need to change these parameters
epsilon = 0.1 
step_size = 0.1 
discount = 0.95


def setN(N):
    global n
    n = N
    return
def setStep_size(a):
    global step_size
    step_size = a
    return

def agent_init():
    global Q,prev_action,prev_y,prev_x,model,n
    """
    Hint: Initialize the variables that need to be reset before each run begins
    Returns: nothing
    """
    
    print("n = %d" %(n))
    
    Q = np.zeros((9,6,len(actions)))
    
    model = np.full((9,6,len(actions),3),10.0)
    
    
    return    

    #initialize the policy array in a smart way

def agent_start(state):
    global Q,prev_action,prev_y,prev_x,model
    """
    Hint: Initialize the variavbles that you want to reset before starting a new episode
    Arguments: state: numpy array
    Returns: action: integer
    """
    # pick the first action, don't forget about exploring starts 
    
    x = state[0]
    y = state[1]
    
    
    action_number = rand_in_range(len(actions))
    
    '''
    if rand_un() < epsilon:
	    action_number = rand_in_range(len(actions))
    else:
	    action_number = np.argmax(Q[x][y]) #find best action
	    if  Q[x][y][action_number] == 0:
		    action_number = rand_in_range(4)
    '''
    
    
    prev_action = action_number
    prev_x = x
    prev_y = y
    
    action = actions[action_number]
    
    return action
	
    


def agent_step(reward, state): # returns NumPy array, reward: floating point, this_observation: NumPy array
    global Q,prev_action,prev_y,prev_x,model
    """
    Arguments: reward: floting point, state: integer
    Returns: action: integer
    """
    # select an action, based on Q
    
    x = state[0] 
    y = state[1]
    
    
    #update Q
    Q[prev_x][prev_y][prev_action] += step_size * (reward+  discount*np.max(Q[x][y]) - Q[prev_x][prev_y][prev_action])
    

    model[(prev_x,prev_y,prev_action)] = [reward,x,y]
    
    
    #pick a state which has been update before
    i = 0
    while i < n:
	    i += 1
	    
	    valid = False
	    while not valid:
		    model_x = rand_in_range(9)
		    model_y = rand_in_range(6)
		    model_action = rand_in_range(4)
		    if model[(model_x,model_y,model_action)][0] != 10.0:
			    valid = True
    
            
	    model_reward = model[(model_x,model_y,model_action)][0]
	    model_nextX = model[(model_x,model_y,model_action)][1]
	    model_nextY = model[(model_x,model_y,model_action)][2]
	    
	    #planning
	    Q[model_x][model_y][model_action] += step_size * (model_reward +discount*np.max(Q[model_nextX][model_nextY]) - Q[model_x][model_y][model_action])
    
    
    #find best action base on epsilon greedy
    if rand_un() < epsilon:
	    action_number = rand_in_range(len(actions))
    else:
	    action_number = np.argmax(Q[x][y]) 
	    if  Q[x][y][action_number] == 0:
		    action_number = rand_in_range(4)
    
    
    prev_x = x
    prev_y = y
    prev_action = action_number
    action = actions[action_number]
	    
    
    return action

def agent_end(reward):
    global  Q,actions,prev_action,prev_y,prev_x,modelDict
    """
    Arguments: reward: floating point
    Returns: Nothing
    """
    # do learning and update pi
    
    Q[prev_x][prev_y][prev_action] += step_size * (reward + 0 - Q[prev_x][prev_y][prev_action] )

    return

def agent_cleanup():
    """
    This function is not used
    """
    # clean up
    return

def agent_message(in_message): # returns string, in_message: string
    global Q
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

