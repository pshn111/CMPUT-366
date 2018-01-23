#!/usr/bin/env python

"""
  Author: Adam White, Matthew Schlegel, Mohammad M. Ajallooeian
  Purpose: for use of Rienforcement learning course University of Alberta Fall 2017
 
  agent does *no* learning, selects actions randomly from the set of legal actions
 
"""

from utils import rand_in_range
import numpy as np
import random

last_action = None # last_action: NumPy array

num_actions = 10

q = 5

epsilon = 0

Q = [0,0,0,0,0,0,0,0,0,0]

step_size = 0.1

def get_greedy(Q):
    old_Q = Q
    Q = sorted(Q,reverse = True)
    choices = []
    for i in range(len(Q)):
        if old_Q[i]==Q[0]:
            choices.append(i)
    #randomly choose from largest estimate actions
    return random.choice(choices)
    
    

def agent_init():
    global last_action

    last_action = np.zeros(1) # generates a NumPy array with size 1 equal to zero

def agent_start(this_observation): # returns NumPy array, this_observation: NumPy array
    global last_action

    local_action = np.zeros(1)
    local_action[0] = rand_in_range(num_actions)
    last_action = local_action
    
    #initialize Q list
    for i in range(10):
        Q[i] = q

    return last_action


def agent_step(reward, this_observation): # returns NumPy array, reward: floating point, this_observation: NumPy array
    global last_action,Q
    local_action = np.zeros(1)
    
    #update Q(At)
    Q[int(last_action[0])] +=  step_size * (reward-Q[int(last_action[0])])
    
    #epsilon==0, we only choose greedy choice
    if epsilon==0:
        local_action[0] = get_greedy(Q)
    else:
        #rand_in_range(10)>0 means 90% agent goes for greedy choice
        if rand_in_range(10)>0:
            #choose the action with argmax Q(a)
            local_action[0] = get_greedy(Q)
        #10% agent choose from random
        else:
            local_action[0] = rand_in_range(num_actions)
        

    last_action = local_action

    return last_action

def agent_end(reward): # reward: floating point
    # final learning update at end of episode
    return

def agent_cleanup():
    # clean up
    return

def agent_message(inMessage): # returns string, inMessage: string
    # might be useful to get information from the agent

    if inMessage == "what is your name?":
        return "my name is skeleton_agent!"
  
    # else
    return "I don't know how to respond to your message"