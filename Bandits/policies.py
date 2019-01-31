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

class OptimisticGreedy(Policy):
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
		
 
class EpsilonGreedy(Greedy):
 	def __init__(self,num_arms,epsilon=0.8):
 		super().__init__(num_arms)
 		self.epsilon=epsilon
 		self.epsilon_min=0.05
 		self.decay_value=0.001

 	def act(self):
 		random_action=np.random.randint(low=0,high=self.num_arms)

 		if np.random.random()>self.epsilon:
 			action = super().act()
 		else:
 			action = random_action
 		if self.epsilon>self.epsilon_min:
 			self.epsilon-=self.decay_value

 		return action



class UCB(Greedy):
	def __init__(self,num_arms):
		super().__init__(num_arms)
		self.timestep=0

	def act(self):
		if self.timestep<self.num_arms:
			action=self.timestep

		else:
			average_rewards=np.divide(self.cumulative_rewards,self.num_pulls,where=self.num_pulls>0)
			ucb_estimates= average_rewards + np.sqrt((2*np.log(self.timestep))/self.num_pulls)
			action=np.argmax(ucb_estimates)
		self.timestep+=1
		return action











