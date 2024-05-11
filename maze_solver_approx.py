import numpy as np
import random
import matplotlib.pyplot as plt

from model import Model


class MazeSolverApprox:
    def __init__(self,maze_generator,CONDITIONS,EPSILON,ALPHA,GAMMA,weights=None,featurizer=None,trained=0):
        self.maze_generator = maze_generator
        self.END_STATE,self.STATE_SPACE,self.ACTION_SPACE,self.MAX_STEPS = CONDITIONS
        self.EPSILON = EPSILON
        self.ALPHA = ALPHA
        self.GAMMA = GAMMA
        self.total_steps=0
        self.cumulative_rewards = []
        self.steps_per_episode_list = []
        self.steps_per_game = []
        self.model = Model(maze_generator,self.STATE_SPACE,self.ACTION_SPACE,self.END_STATE,weights,featurizer,trained)
        if trained==0:
            self.status = 'training'
        if trained==1:
            self.status = 'testing'

    def epsilon_greedy(self,s):
        # we'll use epsilon-soft to ensure all states are visited
        # what happens if you don't do this? i.e. eps=0
        p = np.random.random()
        if p < (1 - self.EPSILON):
            values = self.model.predict_all_actions(s)
            return self.ACTION_SPACE[np.argmax(values)]
        else:
            return np.random.choice(self.ACTION_SPACE)

    def q_learning(self,count_episode,max_steps=20):
        cumulative_rewards_per_episode=0
        steps_per_episode=0
        for count_step in range(max_steps):
            s = self.maze_generator.get_current_state()
            if s==self.END_STATE:
                self.cumulative_rewards.append(cumulative_rewards_per_episode)
                self.steps_per_episode_list.append(steps_per_episode)
                return 1
            a = self.epsilon_greedy(s)
            r = self.maze_generator.get_reward(s,a)
            s2 = self.maze_generator.get_current_state()

            # get the target
            if s2 == self.END_STATE:
                target = r
            else:
                values = self.model.predict_all_actions(s2)
                target = r + self.GAMMA * np.max(values)

            # update the model
            g = self.model.grad(s, a)
            err = target - self.model.predict(s, a)
            self.model.w += self.ALPHA * err * g
            
            print('\n')
            self.maze_generator.print_grid()
            print(f'Status: {self.status} Episode: {count_episode}, Step: {count_step+1}, Old_State: {s}, Action: {a}, Reward: {r}, New_State: {s2}')
            self.total_steps+=1
            cumulative_rewards_per_episode+=r
            steps_per_episode+=1
        self.cumulative_rewards.append(cumulative_rewards_per_episode)
        self.steps_per_episode_list.append(steps_per_episode)
        return 0
    
    def plot_cumulative_rewards(self):
        x = len(self.cumulative_rewards)
        plt.figure(figsize=(10, 5))
        plt.plot(list(range(x)), self.cumulative_rewards, label='Cumulative Reward per Episode')
        plt.xlabel('Episodes')
        plt.ylabel('Cumulative Reward')
        plt.title(f'Cumulative Reward per Episode')
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_steps_per_episode(self):
        x = len(self.steps_per_episode_list)
        plt.figure(figsize=(10, 5))
        plt.plot(list(range(x)), self.steps_per_episode_list, label='Steps per Episode')
        plt.xlabel('Total Episodes')
        plt.ylabel('Steps per Episode')
        plt.title('Steps Required to Solve the Maze in Each Episode')
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_steps_per_game(self):
        x = len(self.steps_per_game)
        plt.figure(figsize=(10, 5))
        plt.plot(list(range(x)), self.steps_per_game, label='Steps per Game')
        plt.xlabel('Total Game')
        plt.ylabel('Steps per Game')
        plt.title('Steps Required to Solve the Maze in Each Game')
        plt.legend()
        plt.grid(True)
        plt.show()

    def start_game(self):
        """
        Loop Episode
        """
        print("\n=============================Start Game==============================\n")
        self.maze_generator.print_grid()
        print('Episode: 0, Step: 0, State: (0,0), Q(s,a): 0')
        self.total_steps=0
        # self.cumulative_rewards=[]
        count_episode = 1
        while True:
            self.maze_generator.i=0
            self.maze_generator.j=0
            isEndState = self.q_learning(count_episode,self.MAX_STEPS)
            if isEndState:
                break
            count_episode+=1
        self.steps_per_game.append(self.total_steps)
        return self.model.w, self.model.featurizer