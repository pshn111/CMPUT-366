#!/usr/bin/env python

"""
  Author: Adam White, Matthew Schlegel, Mohammad M. Ajallooeian
  Purpose: for use of Rienforcement learning course University of Alberta Fall 2017
 
  agent does *no* learning, selects actions randomly from the set of legal actions
 
"""

from utils import rand_in_range,rand_un
import numpy as np
import random

last_action = None # last_action: NumPy array

num_actions = 10

#qArray = [0,0,0,0,0,0,0,0,0,0]
qArray = [5,5,5,5,5,5,5,5,5,5]
epsilon = 0 #you may change it to 0


def agent_init():
    global last_action
    
    
    last_action = np.zeros(1) # generates a NumPy array with size 1 equal to zero

def agent_start(this_observation): # returns NumPy array, this_observation: NumPy array
    global last_action
    
    
    last_action[0] = rand_in_range(num_actions)

    local_action = np.zeros(1)
    local_action[0] = rand_in_range(num_actions)
    return local_action[0]    

    


def agent_step(reward, this_observation): # returns NumPy array, reward: floating point, this_observation: NumPy array
    global last_action,qArray
    
    local_action = np.zeros(1)
    

    lastQ = qArray[int(last_action[0])]  #get new Q value and update qArray
    newQ = lastQ+(0.1)*(reward - lastQ)  #algorithm
    qArray[int(last_action[0])] = newQ
    
    #print(qArray)
    
    if epsilon == 0:
        local_action[0] = getGreedy()
        #print(local_action)
    else:
        if rand_in_range(10)>0:  #90% greedy action
            local_action[0] = getGreedy()
        else:                    #10% random action
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



def getGreedy():   #random select an action which has max value
    global qArray
    maxQ = qArray[0]
    currentAction = []
    n = 0
    for currentQ in qArray:
        if maxQ < currentQ:
            maxQ = currentQ
            currentAction = [n]
        elif maxQ == currentQ:
            currentAction.append(n)
        n+=1
    
    return random.choice(currentAction)

