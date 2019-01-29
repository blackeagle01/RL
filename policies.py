import numpy as np

class Policy(object):
	def __init__(self):
		raise NotImplementedError


	def act(self):
		raise NotImplementedError


	def feedback(self):
		raise NotImplementedError





class Greedy(Policy):
	def __init__(self,num_arms):
		self.num_arms=num_arms
		self.cumulative_rewards= np.zeros(shape=[self.num_arms])
		self.num_pulls=np.zeros(shape=(self.num_arms,))


	def act(self):
		
		average_rewards=np.divide(self.cumulative_rewards,self.num_pulls,where=(self.num_pulls>0))
		action=np.argmax(average_rewards)
		return action

	def feedback(self,action,reward):
		self.cumulative_rewards[action]+=reward
		self.num_pulls[action]+=1

class OptimisticGreedy(object):
	def __init__(self,num_arms):
		self.num_arms=num_arms
		self.average_rewards=np.array([1 for i in range(num_arms)],dtype=np.float)
		self.num_pulls=np.ones(shape=[self.num_arms])


	def act(self):
		action=np.argmax(self.average_rewards)
		return action


	def feedback(self,action,reward):
		self.num_pulls[action]+=1
		
		error=(reward-self.average_rewards[action])/self.num_pulls[action]

		
		self.average_rewards[action] += error
		
       






