#!/usr/bin/env python

"""
  Author: Adam White, Matthew Schlegel, Mohammad M. Ajallooeian, Sina Ghiassian
  Purpose: Skeleton code for Monte Carlo Exploring Starts Control Agent
           for use on A3 of Reinforcement learning course University of Alberta Fall 2017
 
"""

from utils import rand_in_range, rand_un
import numpy as np
import pickle


num_action = 8    #you may need change it
#num_action = 9
epsilon = 0.1     #you may need change it
step_size = 0.5   #you may need change it

Q = None
actions = None
prev_action = None 
prev_x = None 
prev_y = None 


def agent_init():
    global Q,actions,prev_action
    """
    Hint: Initialize the variables that need to be reset before each run begins
    Returns: nothing
    """

    #initialize the policy array in a smart way
    
    Q = np.zeros((10,7,num_action))
    prev_x = 0
    prev_y = 0
    prev_action = 0
    
    if num_action == 9:  #set action dictionary for different actions
	actions = {0:[1,0],1:[1,-1],2:[0,-1],3:[-1,-1],4:[-1,0],5:[-1,1],6:[0,1],7:[1,1],8:[0,0]}
	    
    else:
	actions = {0:[1,0],1:[1,-1],2:[0,-1],3:[-1,-1],4:[-1,0],5:[-1,1],6:[0,1],7:[1,1]}

    return
def agent_start(state):
    global Q,actions,prev_action,prev_y,prev_x
    """
    Hint: Initialize the variavbles that you want to reset before starting a new episode
    Arguments: state: numpy array
    Returns: action: integer
    """
    # pick the first action, don't forget about exploring starts 
    
   
    x = state[0]
    y = state[1]
    
    
    action_number = rand_in_range(num_action)
    
    
    prev_x = x
    prev_y = y
    prev_action = action_number
    action = actions[action_number]    

    return action


def agent_step(reward, state): # returns NumPy array, reward: floating point, this_observation: NumPy array
    global Q,actions,prev_action,prev_y,prev_x
    """
    Arguments: reward: floting point, state: integer
    Returns: action: integer
    """
    # select an action, based on Q
    
    x = state[0] 
    y = state[1]
    
    
    if rand_un() < epsilon:   #exploring
	action_number = rand_in_range(num_action)
    else:
	action_number = np.argmax(Q[x][y]) 
    
    
    #update  Q
    Q[prev_x][prev_y][prev_action] += step_size * (reward+Q[x][y][action_number]-Q[prev_x][prev_y][prev_action])
    
    
    
    prev_x = x
    prev_y = y
    prev_action = action_number
    action = actions[action_number]
	    
    return action

def agent_end(reward):
    global  Q,actions,prev_action,prev_y,prev_x    
    """
    Arguments: reward: floating point
    Returns: Nothing
    """
    # do learning and update pi

    Q[prev_x][prev_y][prev_action] += step_size * (reward - Q[prev_x][prev_y][prev_action])

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

