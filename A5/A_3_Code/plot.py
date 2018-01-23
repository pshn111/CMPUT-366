#!/usr/bin/env python

"""
 Author: Adam White, Matthew Schlegel, Mohammad M. Ajallooeian, Sina Ghiassian, Zach Holland
 Purpose: for use of Rienforcement learning course University of Alberta Fall 2017
"""

import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
	data1 = open("output1.txt", "r")
	data2 = open("output2.txt", "r")
	data3 = open("output3.txt", "r")
	datay1 = []
	datay2 = []
	datay3 = []

	for line in data1:
		line = line.strip()
		datay1.append(line)
	for line in data2:
		line = line.strip()
		datay2.append(line)
	for line in data3:
		line = line.strip()
		datay3.append(line)
	plt.show()

	plt.plot(datay1,label = "n=0")
	plt.plot(datay2,label = "n=5")
	plt.plot(datay3,label = "n=50")
	plt.xlim([1,50])
	plt.xticks([0,10,20,30,40,50])
	plt.ylim(1,800)
	plt.yticks([15,200,400,600,800])
	plt.ylabel('steps per episode')
	plt.xlabel('episode')
	plt.legend()
	plt.show()