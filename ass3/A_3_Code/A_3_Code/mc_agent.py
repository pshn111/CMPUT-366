#!/usr/bin/env python

"""
  Author: Adam White, Matthew Schlegel, Mohammad M. Ajallooeian, Sina Ghiassian
  Purpose: Skeleton code for Monte Carlo Exploring Starts Control Agent
           for use on A3 of Reinforcement learning course University of Alberta Fall 2017
 
"""

from utils import rand_in_range, rand_un
import numpy as np
import pickle

policy = None
Q = None
Reward = np.zeros((99,99))
Num = np.zeros((99,99))
hit = None
epsilon = 0.1 #you may need change it

def agent_init():
    global Q,Num,Reward,epsilon,policy,hit
    """
    Hint: Initialize the variables that need to be reset before each run begins
    Returns: nothing
    """

    #initialize the policy array in a smart way
    
    policy = [min(i,100-i) for i in range(1,100)]
    
   
    Q = np.full((99,99),0.00000000001)
    Reward = np.zeros((99,99))
    Num = np.full((99,99),1) 
    #setPolicy()
    resethit()    

def agent_start(state):
    global Q,Num,Reward,epsilon,policy
    """
    Hint: Initialize the variavbles that you want to reset before starting a new episode
    Arguments: state: numpy array
    Returns: action: integer
    """
    # pick the first action, don't forget about exploring starts 
    
    action = rand_in_range(min(state[0],100-state[0]))+1
    hit[state[0]-1][action-1] += 1
    
	   
    return action


def agent_step(reward, state): # returns NumPy array, reward: floating point, this_observation: NumPy array
    global Q,Num,Reward,epsilon,policy
    """
    Arguments: reward: floting point, state: integer
    Returns: action: integer
    """
    # select an action, based on Q
    action =  policy[state[0]-1]
    
    
    hit[state[0]-1][action-1] += 1
    
    
    return action

def agent_end(reward):
    global Q,Num,Reward,epsilon,policy,hit
    """
    Arguments: reward: floating point
    Returns: Nothing
    """
    # do learning and update pi
    
    
    Reward += (hit*reward)  #add reward
    Num += hit              #add hit number
    resethit()
    Q = Reward/Num         #calculae average
    
    for i in range(99):
        policy[i] = np.argmax(Q[i])+1
    
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
        return pickle.dumps(np.max(Q, axis=1), protocol=0)
    else:
        return "I don't know what to return!!"

def resethit():
    global hit
    hit = np.zeros((99,99))