#!/usr/bin/env python

"""
  Author: Adam White, Mohammad M. Ajallooeian, Sina Ghiassian
  Purpose: Code for the Gambler's problem environment from the Sutton and Barto
  Reinforcement Learning: An Introduction Chapter 4.
  For use in the Reinforcement Learning course, Fall 2017, University of Alberta 
"""

from utils import rand_norm, rand_in_range, rand_un
import numpy as np

wind_list = [0,0,0,1,1,1,2,2,1,0]
start = [0,3]
goal = [7,3]

current_state = None

def env_init():
    global current_state
    current_state = np.zeros(2)


def env_start():
    """ returns numpy array """
    global current_state

    current_state = start
    return current_state

def env_step(action):
    """
    Arguments
    ---------
    action : int
        the action taken by the agent in the current state

    Returns
    -------
    result : dict
        dictionary with keys {reward, state, isTerminal} containing the results
        of the action taken
    """
    global current_state

    
    x = current_state[0]
    y = current_state[1]
    
    
    new_x = current_state[0] + action[0]
    new_y = current_state[1] + action[1]
    
    # update new position if position in the gridworld
    if new_x>=0 and new_x<=9 and new_y>=0 and new_y<=6 :
        current_state = [new_x,new_y]
    
    
    current_state[1]+=wind_list[x]
    #keep position in gridworld after shift upward
    if current_state[1]>6:
        current_state[1] = 6
    
             
    
    is_terminal = (current_state == goal)
    if is_terminal:
        reward = 0
    else:
        reward = -1    

    result = {"reward": reward, "state": current_state, "isTerminal": is_terminal}

    return result

def env_cleanup():
    #
    return

def env_message(in_message): # returns string, in_message: string
    """
    Arguments
    ---------
    inMessage : string
        the message being passed

    Returns
    -------
    string : the response to the message
    """
    return ""
