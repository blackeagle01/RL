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

		plotter=Plotter()
		action_space=np.arange(self.agent.num_arms)
		plotter.plot_bar(action_space, self.agent.num_pulls,xlabel='Arm',ylabel='No of Pulls',title='Arm pulls')
		print('Optimal Arm = {}'.format(self.env.optimal_arm))
		print('Environments Distribution = {}'.format(self.env.arm_probabilities))


class Plotter(object):
	def __init__(self):
		pass


	def plot_bar(self,x,y,xlabel=None,ylabel=None,title=None):
		plt.bar(x,height=y)
		plt.title(title)
		plt.xlabel(xlabel)
		plt.ylabel(ylabel)
		plt.show()

	def plot_curve(self,x,y):
		plt.plot(x,y)
		plt.show()
