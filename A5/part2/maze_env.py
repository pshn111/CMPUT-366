#!/usr/bin/env python

"""
  Author: Adam White, Mohammad M. Ajallooeian, Sina Ghiassian
  Purpose: Code for the Gambler's problem environment from the Sutton and Barto
  Reinforcement Learning: An Introduction Chapter 4.
  For use in the Reinforcement Learning course, Fall 2017, University of Alberta 
"""

from utils import rand_norm, rand_in_range, rand_un
import numpy as np

current_state = None
goal = [8,5]
start = [0,3]

edges = [[2,2],[2,3],[2,4],[5,1],[7,3],[7,4],[7,5]]


def env_init():
    global current_state
    current_state = np.zeros(2)


def env_start():
    """ returns numpy array """
    global current_state

    current_state = start 
    
    return current_state

def env_step(action):
    global current_state
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
    
    x = current_state[0]
    y = current_state[1]



    new_x = x + action[0]
    new_y = y + action[1]


    
    if new_x>=0 and new_x<=8 and new_y>=0 and new_y<=5 :
        if [new_x,new_y] not in edges:
            current_state = [new_x,new_y]


         

    is_terminal = (current_state == goal)
    if is_terminal:
        reward = 1
    else:
        reward = 0



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
