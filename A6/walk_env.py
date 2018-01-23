#!/usr/bin/env python

"""
  Author: Adam White, Mohammad M. Ajallooeian, Sina Ghiassian
  Purpose: Code for the Gambler's problem environment from the Sutton and Barto
  Reinforcement Learning: An Introduction Chapter 4.
  For use in the Reinforcement Learning course, Fall 2017, University of Alberta 
"""
# I modify this file base on assignment 5 maze_env.py

from utils import rand_norm, rand_in_range, rand_un
import numpy as np

current_state = None
rightEdge = 1000
leftEdge = 0
start = 500


def env_init():
    global current_state
    current_state = np.zeros(1)


def env_start():
    """ returns numpy array """
    global current_state

    current_state[0] = start 
    
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
    
    current_state[0] += action

    if current_state[0] <= leftEdge :
        is_terminal = True
        reward = -1

    elif current_state[0] >= rightEdge:
        is_terminal = True
        reward = 1
    else:
        is_terminal = False
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
