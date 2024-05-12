# Reinforcement Learning Project - Maze Solver

This project has demonstrated a maze solver agent that is able to learn how to solve a 7x7 maze based on Q-learning algorithm and feature engineering technique. The agent learns with first two mazes and tested with an unseen third maze. Interestingly, the agent learns really well even with only two mazes. Please continue reading to see the results.

### Training Data

![Train Maze 1](/results/maze1.png "Train Maze 1")
![Train Maze 1 Graphical](/results/maze1_graphical.png "Train Maze 1 Graphical")
![Train Maze 2](/results/maze2.png "Train Maze 2")
![Train Maze 2 Graphical](/results/maze2_graphical.png "Train Maze 2 Graphical")

### Training Performance
#### First Training 
![Steps per Game](/results/train1_game.png "Steps per Game")
![Cumulative Rewards per Episode](/results/train1_reward.png "Cumulative Rewards per Episode")
![Steps per Episode](/results/train1_episode.png "Steps per Episode")

#### Second Training 
![Steps per Game](/results/train2_game.png "Steps per Game")
![Cumulative Rewards per Episode](/results/train2_reward.png "Cumulative Rewards per Episode")
![Steps per Episode](/results/train2_episode.png "Steps per Episode")

### Test Data
![Test Maze](/results/maze3.png "Test Maze ")
![Test Maze Graphical](/results/maze3_graphical.png "Test Maze Graphical")

### Test Performance
![Steps per Game](/results/test1_game.png "Steps per Game")
![Cumulative Rewards per Episode](/results/test1_reward.png "Cumulative Rewards per Episode")
![Steps per Episode](/results/test1_episode.png "Steps per Episode")

### Total Steps per Game (Before Training vs After Training)
The performance of the agent is pretty satisfying as the agent can always solve the maze in one episode for each game even though it's a completely new maze. Compare the difference between the agent's behaviour before and after training:
#### Before Training
![Steps per Game](/results/train1_game.png "Steps per Game")
#### After Training
![Steps per Game](/results/test1_game.png "Steps per Game")

### Statistical Analysis

#### Sample Distribution of Solving Steps per Episode
The statistics illustrated below has shown that the mean steps required for the maze is about 38 steps, where this maze is unseen before. Note that this sample mean is computed using 1000 samples. Thanks to the law of large numbers, this sample mean should be quite close to the true mean. 

This is an interesting results as the optimal steps is only 12 steps. Compared to the performance before training, this improvement is incredibly significant.
![Sample Distribution](/results/sample_distribution.png "Sample Distribution")
![Statistics](/results/statistics.png "Statistics")

#### Distribution of Sample Mean of Solving Steps per Episode
To support our argument for the accuracy of sample mean, we may also utilize the Central Limit Theorem to provide an additional evidence. We define the number of steps required to solve a maze in one episode as the random variable and we have 1000 random variables in each game. Note that the random variables are independent as the agent's behavior in one episode doesn't affect the other episodes during testing period. Furthermore, we may also assume the random variables are identically distributed since the agent keeps following the same policy for every episode during testing period. Based on the condition of Independent and Identically Distributed (IID) random variables, the sample means of each game will approximately form a normal distribution if the number of games is large enough.

Indeed, we have proved that with the illustration as shown below. The mean steps required to solve a maze per episode is indeed about 38 steps and this argument is supported by the 95% confidence interval.
![Sample Mean Distribution](/results/sample_mean_distribution.png "Sample Mean Distribution")
![Confidence Interval](/results/confidence_interval.png "Confidence Interval")


## How to Run
### Training
To train the agent, run the following command. The agent will be tested automatically as well after training with this command. The weights and featurizer will be saved in the /configs folder after training.
```
python main.py train
```

### Testing
To test the agent only without updating the weights and featurizer, run the following command.
```
python main.py test
```

## About Author
Hi, my name is Lor Wen Sin (Albert). I'm currently studying Mechanical Engieering (Robotics stream) at Nanyang Technological University Singapore. My research interest is about Human-Robot Collaboration and I'm currently looking for an opportunity to further my studies in the United States. 

You are allowed to use this project as you like. Also, feel free to contact me for further discussion if you are interested.

## Contact
[![Email](https://img.shields.io/badge/Email-contact-blue)](mailto:WLOR001@e.ntu.edu.sg)
[![Email](https://img.shields.io/badge/Email-contact-blue)](mailto:wensinlor@gmail.com)
[LinkedIn](https://www.linkedin.com/in/wen-sin-lor-455ab6232/)
[GoogleScholar](https://scholar.google.com/citations?user=K1SD2oUAAAAJ&hl=zh-CN)