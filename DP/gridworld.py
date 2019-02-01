#import numpy as np











class Environment(object):
    def __init__(self):
        current_state=None
        terminal_state=None
        start_state=None
        episode_over=False
        state_transition_probabilities=None
        
        
    def step(action):
        pass
    
    
    
    def reset():
        pass

class Gridworld(Environment):
    def __init__(self,n,start_state=None):
        
        self.state_space=[(i,j) for i in range(n) for j in range(n)]
        self.action_space=['up','down','right','left']
        self.min=0
        self.max=n-1
        self.current_state=None
        self.episode_over=False
        self.terminal_states=[(0,0),(n-1,n-1)]
        if start_state is None:
            self.start_state=(n/2,n/2)
            
        else:
            self.start_state=start_state
        
    def transition(self,state,action):
        if state in self.terminal_states:
            return state
        i,j=state
        if action=='up':
            if i==self.min:
                return (i,j)
            else:
                return (i-1,j)
            
        if action=='down':
            if i==self.max:
                return (i,j)
            else:
                return (i+1,j)
            
        if action=='left':
            if j==self.min:
                return (i,j)
            else:
                return (i,j-1)
            
        if action=='right':
            if j==self.max:
                return (i,j)
            else:
                return (i,j+1)
            
            
            
    def step(self,action):
        x=self.current_state
        newstate=self.transition(self.current_state,action)
        
        
        if newstate in self.terminal_states:
            self.episode_over=True
        self.current_state=newstate





        if x in self.terminal_states:
            reward=0
            
        else:
            reward=-1
            
        return newstate,reward
    
    
    def reset(self):
        self.current_state=self.start_state
        self.episode_over=False
                
            
            
        
        
        
        