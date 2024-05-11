import random
import matplotlib.pyplot as plt


class MazeSolver:
    def __init__(self,maze_generator,CONDITIONS,EPSILON,ALPHA,GAMMA,Q):
        self.maze_generator = maze_generator
        self.END_STATE,self.STATE_SPACE,self.ACTION_SPACE,self.MAX_STEPS = CONDITIONS
        self.EPSILON = EPSILON
        self.ALPHA = ALPHA
        self.GAMMA = GAMMA
        self.Q = Q
        self.total_steps=0
        self.cumulative_rewards=[]
        self.steps_per_game=[]

    def maxQ(self,s):
        max_value = max(self.Q[s].values())
        optimal_actions = [key for key, value in self.Q[s].items() if value == max_value]
        return random.choice(optimal_actions)

    def epsilon_greedy(self,s):
        p = random.random()
        if p < 1-self.EPSILON:
            return self.maxQ(s)
        else:
            return random.choice(self.ACTION_SPACE)

    def q_learning(self,count_episode,max_steps=20):
        cumulative_rewards_per_episode=0
        for count_step in range(max_steps):
            s = self.maze_generator.get_current_state()
            if s==self.END_STATE:
                self.cumulative_rewards.append(cumulative_rewards_per_episode)
                return 1
            a = self.epsilon_greedy(s)
            r = self.maze_generator.get_reward(s,a)
            s2 = self.maze_generator.get_current_state()
            opt_a = self.maxQ(s2)
            self.Q[s][a] = self.Q[s][a] + self.ALPHA*(r + self.GAMMA*self.Q[s2][opt_a] - self.Q[s][a])
            print('\n')
            self.maze_generator.print_grid()
            print(f'Episode: {count_episode}, Step: {count_step+1}, Old_State: {s}, Action: {a}, Reward: {r}, New_State: {s2}, Q(s,a): {self.Q[s][a]}')
            self.total_steps+=1
            cumulative_rewards_per_episode+=r
        self.cumulative_rewards.append(cumulative_rewards_per_episode)
        return 0
    
    def plot_cumulative_rewards(self,i):
        x = len(self.cumulative_rewards)
        plt.figure(figsize=(10, 5))
        plt.plot(list(range(x)), self.cumulative_rewards, label='Cumulative Reward per Episode')
        plt.xlabel('Episodes')
        plt.ylabel('Cumulative Reward')
        plt.title(f'Cumulative Reward per Episode Over Time - Game {i+1}')
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
        self.cumulative_rewards=[]
        count_episode=1
        while True:
            self.maze_generator.i=0
            self.maze_generator.j=0
            isEndState = self.q_learning(count_episode,self.MAX_STEPS)
            if isEndState:
                break
            count_episode+=1
        #self.plot_cumulative_rewards(i)
        self.steps_per_game.append(self.total_steps)