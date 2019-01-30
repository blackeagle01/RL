import numpy as np 
import matplotlib.pyplot as plt 



class Simulator(object):

	def __init__(self,agent,environment):
		self.agent=agent
		self.env=environment
		

	def run_simulation(self,num_trials):
		regret_log=list()

		for x in range(num_trials):
			action=self.agent.act()
			
			reward,regret=self.env.step(action)
			regret_log.append(regret)
			
			self.agent.feedback(action,reward)

		print('Simulation Complete')

		regret_log=np.array(regret_log,dtype=np.float)
		cumulative_regret=np.cumsum(regret_log)
		average_regret=np.mean(cumulative_regret)







		plotter=Plotter()
		action_space=np.arange(self.agent.num_arms)
		plotter.plot_bar(action_space, self.agent.num_pulls,xlabel='Arm',ylabel='No of Pulls',title='Arm pulls')
		
		print('Optimal Arm = {}'.format(self.env.optimal_arm))
		print('Average Regret = {}'.format(average_regret))
		print('Environments Distribution = {}'.format(self.env.arm_probabilities))
		trial_axis=np.arange(num_trials)


		plotter.plot_curve(trial_axis,cumulative_regret,xlabel='time',ylabel='regret',title='Regret over time')



class Plotter(object):
	def __init__(self):
		pass


	def plot_bar(self,x,y,xlabel=None,ylabel=None,title=None):
		plt.bar(x,height=y)
		plt.title(title)
		plt.xlabel(xlabel)
		plt.ylabel(ylabel)
		plt.show()

	def plot_curve(self,x,y,xlabel=None,ylabel=None,title=None):
		plt.plot(x,y)
		plt.title(title)
		plt.xlabel(xlabel)
		plt.ylabel(ylabel)
		plt.show()

	
