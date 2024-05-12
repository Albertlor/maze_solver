import random


class MazeGenerator:
    TEST_MAZE_STRUCTURE = {
        (0,0):['D','R'],
        (0,1):['L','R'],
        (0,2):['L','R','D'],
        (0,3):['L','R'],
        (0,4):['L','D'],
        (1,0):['U','D'],
        (1,2):['U','D'],
        (1,4):['U','R'],
        (1,5):['L','D','R'],
        (1,6):['L','D'],
        (2,0):['U','D'],
        (2,2):['U','D'],
        (2,5):['U','D','R'],
        (2,6):['U','D','L'],
        (3,0):['U','R'],
        (3,1):['L','R'],
        (3,2):['L','R'],
        (3,3):['L','D','R'],
        (3,4):['L','R'],
        (3,5):['U','L','R'],
        (3,6):['U','L'],
        (4,3):['U','D'],
        (5,3):['U','D','R'],
        (5,4):['L','R'],
        (5,5):['L','R'],
        (5,6):['L'],
        (6,0):['R'],
        (6,1):['L','R'],
        (6,2):['L','R'],
        (6,3):['U','L']
    }
    TRAINING_MAZE_STRUCTURE = {
        (0,0):['D','R'],
        (0,1):['L','R'],
        (0,2):['L','D'],
        (0,4):['D','R'],
        (0,5):['L','R'],
        (0,6):['L','D'],
        (1,0):['U','D'],
        (1,2):['U','D'],
        (1,4):['U','D'],
        (1,6):['U','D'],
        (2,0):['U','D'],
        (2,2):['U','D'],
        (2,4):['U','D'],
        (2,6):['U','D'],
        (3,0):['U','R'],
        (3,1):['L','R'],
        (3,2):['U','L','R'],
        (3,3):['L','D','R'],
        (3,4):['U','L','R'],
        (3,5):['L','R'],
        (3,6):['U','L'],
        (4,3):['U','D'],
        (5,2):['D','R'],
        (5,3):['U','L','R'],
        (5,4):['L','R'],
        (5,5):['L','R'],
        (5,6):['L'],
        (6,0):['R'],
        (6,1):['L','R'],
        (6,2):['U','L']
    }

    def __init__(self,rows,columns,maze_structure='training_maze'):
        self.rows = rows
        self.columns = columns
        self.i = 0
        self.j = 0
        if maze_structure=='training_maze':
            self.maze_structure = MazeGenerator.TRAINING_MAZE_STRUCTURE
        if maze_structure=='test_maze':
            self.maze_structure = MazeGenerator.TEST_MAZE_STRUCTURE

    def generate_maze(self,STEP_COST,PENALTY_COST,PRIZE,CONSTRAINTS):
        self.STEP_COST = STEP_COST
        self.PENALTY_COST = PENALTY_COST
        self.PRIZE = PRIZE
        self.START_STATE,self.END_STATE,self.END_CONDITION,self.ACTION_SPACE = CONSTRAINTS
        
        directions = {
            'U': (-1, 0),
            'D': (1, 0),
            'L': (0, -1),
            'R': (0, 1)
        }

        # Initialize the maze structure with only the end state
        maze_structure = {self.END_STATE: []}
        
        # Generate a path from start to end
        path = []
        current = self.START_STATE
        while current != self.END_STATE:
            if current not in path:
                path.append(current)
            # Determine feasible actions leading closer to the end state
            possible_actions = []
            for action, (dr, dc) in directions.items():
                next_state = (current[0] + dr, current[1] + dc)
                if 0 <= next_state[0] < self.rows and 0 <= next_state[1] < self.columns:
                    if next_state not in path:  # Prevent looping back to already included states
                        possible_actions.append((action, next_state))
            if not possible_actions:
                break  # Exit if no further move is possible
            selected_action, next_state = random.choice(possible_actions)
            if current not in maze_structure:
                maze_structure[current] = []
            maze_structure[current].append(selected_action)
            current = next_state
        
        # Add actions for last state if path is successful
        if current == self.END_STATE and current not in maze_structure:
            maze_structure[current] = []
            if current in path and len(path) > 1:
                prev_state = path[-2]
                for action, (dr, dc) in directions.items():
                    if (prev_state[0] + dr, prev_state[1] + dc) == current:
                        maze_structure[current].append(action)

        return maze_structure
        

    def update_state(self,a):
        if a=='U':
            self.i-=1
        elif a=='L':
            self.j-=1
        elif a=='D':
            self.i+=1
        elif a=='R':
            self.j+=1

    def get_reward(self,s,a):
        available_actions = self.maze_structure[s]
        end_condition_s = list(self.END_CONDITION.keys())
        end_condition_a = list(self.END_CONDITION.values())
        if s==self.END_STATE:
            return 0
        else:
            if s==end_condition_s[0] and a==end_condition_a[0]:
                self.update_state(a)
                return self.PRIZE
            elif a in available_actions:
                self.update_state(a)
                return self.STEP_COST 
            else:
                return self.PENALTY_COST   
            
    def get_current_state(self):
        return (self.i, self.j)

    def print_grid(self):
        path_list = list(self.maze_structure.keys())
        for r in range(self.rows):
            print('| ',end="")
            for c in range(self.columns):
                if (r,c) not in path_list:
                    print('1 | ',end="")
                else:
                    if (r,c) == (self.i,self.j):
                        print('* | ',end="")
                    else:
                        print('0 | ',end="")
            print('\n')