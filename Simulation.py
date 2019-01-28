import numpy as np 
import matplotlib.pyplot as plt 



class Simulator(object):

	def __init__(self,agent,environment):
		self.agent=agent
		self.env=environment

	def run_simulation(self,num_trials):

		for x in range(num_trials):
			action=self.agent.act()
			reward,regret=self.env.step(action)
			self.agent.feedback(action,reward)

		print('Simulation Complete')

		print(self.agent.num_pulls)