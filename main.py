import numpy as np
import sys

from maze_generator import MazeGenerator
from maze_solver import MazeSolver
from maze_solver_approx import MazeSolverApprox
from joblib import dump, load


def train_agent(maze_generator):
    END_STATE = maze_generator.END_STATE
    STATE_SPACE = maze_generator.maze_structure.keys()
    ACTION_SPACE = ['U','L','D','R']
    MAX_STEPS = 40
    CONDITIONS = [END_STATE,STATE_SPACE,ACTION_SPACE,MAX_STEPS]

    """
    Initialize Q value
    """
    Q={}
    for s in STATE_SPACE:
        Q[s]={}
        for a in ACTION_SPACE:
            Q[s][a]=0
    
    EPSILON = 0.2
    ALPHA = 0.1
    GAMMA = 0.9
    maze_solver = MazeSolver(maze_generator,CONDITIONS,EPSILON,ALPHA,GAMMA,Q)
    for _ in range(100):
        maze_solver.start_game()
        maze_generator.i=0
        maze_generator.j=0
    maze_solver.plot_steps_per_game()

    return maze_solver.Q

def test_agent(maze_generator,Q):
    END_STATE = maze_generator.END_STATE
    STATE_SPACE = maze_generator.maze_structure.keys()
    ACTION_SPACE = ['U','L','D','R']
    MAX_STEPS = 40
    CONDITIONS = [END_STATE,STATE_SPACE,ACTION_SPACE,MAX_STEPS]

    """
    Adapt Q value for new environment based on the trained Q
    """
    for s in STATE_SPACE:
        if s not in Q:
            Q[s]={}
        for a in ACTION_SPACE:
            if a not in Q[s]:
                Q[s][a]=0

    EPSILON = 0.1
    ALPHA = 0.1
    GAMMA = 0.9
    maze_solver = MazeSolver(maze_generator,CONDITIONS,EPSILON,ALPHA,GAMMA,Q)
    for _ in range(100):
        maze_solver.start_game()
        maze_generator.i=0
        maze_generator.j=0
    maze_solver.plot_steps_per_game()

    return maze_solver.Q

def train_agent_approx(maze_generator):
    END_STATE = maze_generator.END_STATE
    STATE_SPACE = maze_generator.maze_structure.keys()
    ACTION_SPACE = ['U','L','D','R']
    MAX_STEPS = 200
    CONDITIONS = [END_STATE,STATE_SPACE,ACTION_SPACE,MAX_STEPS]
    
    EPSILON = 0.2
    ALPHA = 0.05
    GAMMA = 0.9
    maze_solver_approx = MazeSolverApprox(maze_generator,CONDITIONS,EPSILON,ALPHA,GAMMA)
    for _ in range(1000):
        weights, featurizer = maze_solver_approx.start_game()
        maze_generator.i=0
        maze_generator.j=0
    maze_solver_approx.plot_steps_per_game()
    maze_solver_approx.plot_cumulative_rewards()
    maze_solver_approx.plot_steps_per_episode()

    return weights, featurizer

def test_agent_approx(maze_generator,weights,featurizer):
    END_STATE = maze_generator.END_STATE
    STATE_SPACE = maze_generator.maze_structure.keys()
    ACTION_SPACE = ['U','L','D','R']
    MAX_STEPS = 200
    CONDITIONS = [END_STATE,STATE_SPACE,ACTION_SPACE,MAX_STEPS]

    EPSILON = 0.2
    ALPHA = 0.05
    GAMMA = 0.9
    maze_solver_approx = MazeSolverApprox(maze_generator,CONDITIONS,EPSILON,ALPHA,GAMMA,weights=weights,featurizer=featurizer,trained=1)
    for _ in range(1000):
        maze_solver_approx.start_game()
        maze_generator.i=0
        maze_generator.j=0
    maze_solver_approx.plot_steps_per_game()
    maze_solver_approx.plot_cumulative_rewards()
    maze_solver_approx.plot_steps_per_episode()

    return weights


ACTION_SPACE = ['U','L','D','R']
status = sys.argv[1]
if status == 'train':
    """
    Training
    """
    ######################################
    STEP_COST = -1
    PENALTY_COST = -10
    PRIZE = 100
    START_STATE = (0,0)
    END_STATE = (6,0)
    END_CONDITION = {(6,1):'L'}
    MAZE_CONSTRAINTS = [START_STATE,END_STATE,END_CONDITION,ACTION_SPACE]
    maze_generator = MazeGenerator(7,7,'training_maze')
    maze_generator.generate_maze(STEP_COST,PENALTY_COST,PRIZE,MAZE_CONSTRAINTS)
    #trained_Q = train_agent(maze_generator)
    trained_weights, trained_featurizer = train_agent_approx(maze_generator)
    np.save('./configs/weights1.npy', trained_weights)
    dump(trained_featurizer, './configs/featurizer1.joblib')
    status = 'test'
    ######################################

if status == 'test':
    trained_weights = np.load('./configs/weights1.npy')
    trained_featurizer = load('./configs/featurizer1.joblib')
    """
    Test
    """
    ######################################
    STEP_COST = -1
    PENALTY_COST = -10
    PRIZE = 100
    START_STATE = (0,0)
    END_STATE = (6,0)
    END_CONDITION = {(6,1):'L'}
    MAZE_CONSTRAINTS = [START_STATE,END_STATE,END_CONDITION,ACTION_SPACE]
    maze_generator = MazeGenerator(7,7,'test_maze')
    maze_generator.generate_maze(STEP_COST,PENALTY_COST,PRIZE,MAZE_CONSTRAINTS)
    # tested_Q = test_agent(maze_generator,trained_Q)
    tested_weights = test_agent_approx(maze_generator,trained_weights,trained_featurizer)
    ######################################