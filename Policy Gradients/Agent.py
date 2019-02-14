import numpy as np
from policy import PolicyNetwork
import torch
from torch.nn import functional as F

class Agent(object):

	def __init__(self,env,gamma=0.98,learning_rate=0.01):
		self.env = env
		self.net = PolicyNetwork()
		self.optimiser = torch.optim.Adam(self.net.parameters(),lr=learning_rate)
		self.gamma = gamma


	def learn(self,num_episodes=1000):

		count=0

		for episode in range(num_episodes):

			experiences = self.collect_experiences()

			states,actions,rewards,total_rewards = experiences

			predicted_action_distribution = self.net(states)

			neg_log_loss = F.cross_entropy(predicted_action_distribution,actions,reduce=False)

			loss = torch.mean(neg_log_loss * rewards)

			self.optimiser.zero_grad()
			loss.backward()
			self.optimiser.step()

			#if episode%20:
			#	print('Total Rewards = {} '.format(total_rewards))

			if total_rewards==self.env._max_episode_steps:
				count+=1

			else:
				count=0

			if count==6:
				print("Learnt")
				break

		return self.net





	def _sample_action(self,action_probs):

		data = action_probs.data.numpy()
		data = data.ravel()

		action = np.random.choice(range(len(data)),p=data)

		return action


	def _discount_rewards(self,rewards):
		rewards = np.array(rewards)

		discounted_rewards = np.zeros_like(rewards)
    
		for i in range(len(rewards)):
			cumsum=0
			for j in range(i,len(rewards)):
				cumsum+= self.gamma**(j-i) * rewards[j]
        
			discounted_rewards[i]=cumsum
       
		mean=discounted_rewards.mean()
		std=discounted_rewards.std()
	    
		discounted_rewards-=mean
		discounted_rewards/=std
	        
		return discounted_rewards






	def collect_experiences(self):

		done = False
		state= self.env.reset()

		statebuffer,actionbuffer,rewardbuffer=[],[],[]

		while not done:
			statebuffer.append(state)
			state = torch.from_numpy(state).unsqueeze(0).float()
			action_probs = self.net(state)
			action = self._sample_action(action_probs)
			actionbuffer.append(action)
			newstate,reward,done,_ = self.env.step(action)
			rewardbuffer.append(reward)
			state=newstate

		total_rewards = np.array(rewardbuffer).sum()

		statebuffer = np.array(statebuffer)
		actionbuffer = np.array(actionbuffer)



		rewards = self._discount_rewards(rewardbuffer)
		states = torch.from_numpy(statebuffer).float()
		actions = torch.from_numpy(actionbuffer).long()
		rewards = torch.from_numpy(rewards).float()

		return (states,actions,rewards,total_rewards)



	def simulate(self,num_episodes=10):
		print('Simulating for {} episodes'.format(num_episodes))

		for episode in range(num_episodes):
			done=False
			state=self.env.reset()
			total_reward=0
			while not done:
				state = torch.from_numpy(state).unsqueeze(0).float()
				action = self.net(state)
				action = np.argmax(action.data.numpy())

				newstate,reward,done,_ =self.env.step(action)
				state= newstate
				total_reward+=reward
				self.env.render()
		

			print("Reward for episode {} = {}".format(episode,total_reward))

		self.env.close()






