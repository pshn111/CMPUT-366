#!/usr/bin/env python

"""
 Author: Adam White, Matthew Schlegel, Mohammad M. Ajallooeian, Sina Ghiassian, Zach Holland
 Purpose: for use of Rienforcement learning course University of Alberta Fall 2017
"""

import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
   V = open('ValueFunction.txt', "r")
   x_list=[]
   y_list=[]
   plt.show()
   
   for line in V:
      line = line.strip()
      x = int(line.split(",")[0])
      x_list.append(x)
      y = int(line.split(",")[1])
      y_list.append(y)      
      
  
   plt.plot(x_list,y_list)
   plt.xlim([0,8000])
   plt.xticks([0,1000,2000,3000,4000,5000,6000,7000,8000])
   plt.xlabel('Time steps')
   plt.ylabel('Episodes')
   plt.legend()
   plt.show()