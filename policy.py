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





	


if __name__ == '__main__':

	g=Gridworld(4)

	state_space=g.state_space
	action_space=g.action_space
	
	p=Policy(state_space, action_space)

	print(p.evaluate(g)) 




