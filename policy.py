#import numpy as np 
import random
from gridworld import Gridworld

class Policy(object):
	def __init__(self,state_space,action_space):
		self.state_space=state_space
		self.action_space=action_space

		self.pi= {s : {a:0.25 for a in action_space} for s in state_space}



	def __getitem__(self,state):
		return random.choice(self.pi[state].items())



	def evaluate(self,env,theta=0.01,gamma=0.8):

		env.reset()
		delta=100

		V={s:0 for s in self.state_space}

		while delta>theta:
			delta=0
			for state in state_space:
				val=0
				v=V[state]
				for action in action_space:
					env.current_state=state
					newstate,reward= env.step(action)
					val += self.pi[state][action]*(reward + gamma * V[newstate])
				V[state]=val
				delta = max(delta,abs(v-V[state]))

		return V



	def value_iteration(self,env,theta=0.01,gamma=0.8):

		delta=100
		V={s:0 for s in self.state_space}

		while delta>theta:
			delta=0
			for state in state_space:
				action_values=[]
				v=V[state]
				for action in action_space:
					env.current_state=state 
					newstate,reward=env.step(action)
					action_values.append(reward + gamma * V[newstate])
				V[state]=max(action_values)
				delta= max(delta,abs(v-V[state]))

		return V



	def generate_policy(self,env):
		V=self.value_iteration(env)

		for state in self.state_space:
			q=[]
			for action in action_space:
				env.current_state=state
				newstate,reward=env.step(action)
				action_value=reward + 0.8 * V[newstate]
				q.append([action,action_value])
			self.pi[state] = max(q,key=self._argmax)[0]



	def navigate_home(self,env,current_state):
		self.generate_policy(env)
		env.current_state=current_state

		prev_state=None
		while prev_state!=current_state:
			action=self.pi[current_state]
			newstate,reward=env.step(action)
			print(action,newstate)
			prev_state=current_state
			current_state=newstate







	def _argmax(self,x):
		return x[1]






	


if __name__ == '__main__':

	g=Gridworld(10)

	state_space=g.state_space
	action_space=g.action_space
	
	p=Policy(state_space, action_space)


	p.navigate_home(g,current_state=(5,8))




