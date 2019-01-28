import numpy as np


class Environment(object):
	def __init__(self):
		pass

	def step():
		raise NotImplementedError('Abstract class, cant create object')




class Bandit(Environment):
	def __init__(self,num_arms,distribution='bernoulli'):
		self.num_arms=num_arms
		self.distribution=distribution

		if self.distribution=='bernoulli':
			self.arm_probabilities=np.random.random(size=[self.num_arms])

		self.optimal_arm=np.max(self.arm_probabilities)

	def step(self,arm): 
		regret = self.optimal_arm-arm

		if np.random.random()>self.arm_probabilities[arm]:
			return 0,regret

		else:
			return 1,regret
