# ✅ Maze + A* + Trajectory Generation Setup
# %%
import numpy as np

import heapq
import copy
import random

import torch
from torch.utils.data import Dataset, DataLoader

# import gymnasium as gym
# import gymnasium_robotics

# gym.register_envs(gymnasium_robotics)

# from stable_baselines3.common.vec_env import DummyVecEnv

# %%
# ----- A* Pathfinding over a binary occupancy grid -----
def astar(grid, start, goal):
    def heuristic(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    neighbors = [(0,1), (1,0), (-1,0), (0,-1)]
    close_set = set()
    came_from = {}
    gscore = {start:0}
    fscore = {start:heuristic(start, goal)}
    oheap = []

    heapq.heappush(oheap, (fscore[start], start))

    while oheap:
        _, current = heapq.heappop(oheap)
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()
            return path

        close_set.add(current)
        for dx, dy in neighbors:
            neighbor = (current[0]+dx, current[1]+dy)
            tentative_g_score = gscore[current] + 1
            if 0 <= neighbor[0] < grid.shape[0] and 0 <= neighbor[1] < grid.shape[1]:
                if grid[neighbor] == 1:
                    continue
            else:
                continue

            if neighbor in close_set and tentative_g_score >= gscore.get(neighbor, 0):
                continue

            if tentative_g_score < gscore.get(neighbor, float('inf')):
                came_from[neighbor] = current
                gscore[neighbor] = tentative_g_score
                fscore[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                heapq.heappush(oheap, (fscore[neighbor], neighbor))

    return []  # No path found


# ----- Interpolate Grid Path into Continuous Trajectory -----
def interpolate_path(path, num_points=10):
    if len(path) < 2:
        return np.array(path)
    path = np.array(path)
    distance = np.cumsum(np.sqrt(np.sum(np.diff(path, axis=0)**2, axis=1)))
    distance = np.insert(distance, 0, 0)
    interpolated = np.zeros((num_points, 2))
    for i in range(2):
        interpolated[:, i] = np.interp(
            np.linspace(0, distance[-1], num_points),
            distance, path[:, i]
        )
    return interpolated


# ----- Build maze structure from symbolic map -----
def parse_maze_map(example_map):
    symbol_to_int = {"0": 0, 0: 0, "1": 1, 1: 1, "g": 0, "r": 2, "c": 0}
    maze = np.array([[symbol_to_int[cell] for cell in row] for row in example_map])
    return maze


# ----- Visualize a Maze and Trajectory -----
def plot_maze_traj(maze, traj, start, goal):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib is required for plotting. Please install it.")
    return
    
    plt.imshow(maze.T, origin='lower', cmap='gray_r')
    plt.plot(traj[:, 0], traj[:, 1], marker='o', color='blue')
    
    # if start:
    #     plt.plot(start[0], start[1], marker='o', color='green', markersize=10, label='start')
    # if goal:
    #     plt.plot(goal[0], goal[1], marker='*', color='yellow', markersize=10, label='goal')

    if start is not None:
        plt.plot(start[0], start[1], marker='s', color='green', markersize=10, label='start')
    if goal is not None:
        plt.plot(goal[0], goal[1], marker='*', color='yellow', markersize=12, label='goal')
    
    plt.title("Shortest Trajectory in Maze")
    plt.grid(True)
    plt.legend()
    plt.show()
    

def find_symbol_position(maze_map, symbol):
    for i, row in enumerate(maze_map):
        for j, cell in enumerate(row):
            if cell == symbol:
                return (i, j)
    return None  # not found


def generate_random_maze_with_start_goal(base_maze, seed=None):
    if seed is not None:
        random.seed(seed)

    # Deep copy to avoid mutating the base
    maze_copy = copy.deepcopy(base_maze)

    # Collect all free positions (value == 0)
    free_positions = [(i, j) for i, row in enumerate(maze_copy)
                      for j, cell in enumerate(row) if cell == 0]

    if len(free_positions) < 2:
        raise ValueError("Not enough free space to place start and goal.")

    # Choose two distinct positions
    start_pos, goal_pos = random.sample(free_positions, 2)

    # Mark them in the copied maze
    maze_copy[start_pos[0]][start_pos[1]] = "r"  # start/reset
    maze_copy[goal_pos[0]][goal_pos[1]] = "g"    # goal

    return maze_copy, start_pos, goal_pos


# Example usage
if __name__ == "__main__":
    
    C = "c"  # Stands for combined cell and indicates that this cell can be initialized as a goal or agent reset location.
    G = "g"  # goal
    R = "r"  # reset
    U_MAZE = [[1, 1, 1, 1, 1],
        [1, 0, 0, 0, 1],
        [1, 1, 1, 0, 1],
        [1, 0, 0, 0, 1],
        [1, 1, 1, 1, 1]]
    
    MEDIUM_MAZE  = [[1, 1, 1, 1, 1, 1, 1, 1],
            [1, 0, 0, 1, 1, 0, 0, 1],
            [1, 0, 0, 1, 0, 0, 0, 1],
            [1, 1, 0, 0, 0, 1, 1, 1],
            [1, 0, 0, 1, 0, 0, 0, 1],
            [1, 0, 1, 0, 0, 1, 0, 1],
            [1, 0, 0, 0, 1, 0, 0, 1],
            [1, 1, 1, 1, 1, 1, 1, 1]]
    
    LARGE_MAZE = [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
                [1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1],
                [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
                [1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1],
                [1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1],
                [1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1],
                [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]
    
    base = U_MAZE
    # base = MEDIUM_MAZE
    # base = LARGE_MAZE
    random_maze, start, goal = generate_random_maze_with_start_goal(base)
    
    print('random_maze:', random_maze)
    print('start:', start)
    print('goal:', goal)

    # Parse for planning (1 = wall, 0 = free)
    maze = parse_maze_map(random_maze)
    print('maze:', type(maze), maze.shape, maze)

    # Define start and goal (from symbolic map)
    # get 2D coordinates of start and goal
    
    start = find_symbol_position(random_maze, R)
    goal = find_symbol_position(random_maze, G)
    # start = (1,6)
    # goal = (6, 1)
    # start = (5, 4)
    # goal = (7, 2)
    print('start:', start, 'goal:', goal)

    # A* and interpolation
    path = astar(maze, start, goal)
    traj = interpolate_path(path, num_points=20)
    print('traj:', traj)

    # Plot
    plot_maze_traj(maze, traj, start, goal)

    # # Optionally: launch env with same map
    # env = gym.make("PointMaze_UMaze-v3", maze_map=example_map)
    # env.reset()
    # env.render()
# %%
# ----- Dataset class for random mazes and trajectories -----
class MazeTrajectoryDataset(Dataset):
    def __init__(self, num_trajectories=1000, num_points=10, templates=None):
        self.num_trajectories = num_trajectories
        self.num_points = num_points
        
        # U_MAZE = [[1, 1, 1, 1, 1],
        #     [1, 0, 0, 0, 1],
        #     [1, 1, 1, 0, 1],
        #     [1, 0, 0, 0, 1],
        #     [1, 1, 1, 1, 1]]
        
        MEDIUM_MAZE  = [[1, 1, 1, 1, 1, 1, 1, 1],
                [1, 0, 0, 1, 1, 0, 0, 1],
                [1, 0, 0, 1, 0, 0, 0, 1],
                [1, 1, 0, 0, 0, 1, 1, 1],
                [1, 0, 0, 1, 0, 0, 0, 1],
                [1, 0, 1, 0, 0, 1, 0, 1],
                [1, 0, 0, 0, 1, 0, 0, 1],
                [1, 1, 1, 1, 1, 1, 1, 1]]
        
        # LARGE_MAZE = [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        #             [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
        #             [1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        #             [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
        #             [1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1],
        #             [1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1],
        #             [1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1],
        #             [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
        #             [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]
        
        self.templates = [
            # U_MAZE,
            MEDIUM_MAZE,
            # LARGE_MAZE
        ]
        
        if templates is not None:
            self.templates = templates

    def __len__(self):
        return self.num_trajectories

    def __getitem__(self, idx):
        base_maze = random.choice(self.templates)
        maze_map, start, goal = generate_random_maze_with_start_goal(base_maze)
        maze = parse_maze_map(maze_map)
        path = astar(maze, start, goal)
        traj = interpolate_path(path, num_points=self.num_points)
        return {
            'maze': torch.tensor(maze, dtype=torch.float32), # [b, h, w]
            'trajectory': torch.tensor(traj, dtype=torch.float32), # [b, num_points, 2]
            'start': torch.tensor(start, dtype=torch.long), # [b, 2]
            'goal': torch.tensor(goal, dtype=torch.long), # [b, 2]
        }


# Example usage
if __name__ == "__main__":
    BASE_MAZE = [
        [1, 1, 1, 1, 1],
        [1, 0, 0, 0, 1],
        [1, 0, 1, 0, 1],
        [1, 0, 0, 0, 1],
        [1, 1, 1, 1, 1],
    ]

    dataset = MazeTrajectoryDataset(num_trajectories=100, num_points=10)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    for batch in dataloader:
        maze = batch['maze'][0].numpy()
        traj = batch['trajectory'][0].numpy()
        start = batch['start'][0].numpy()
        goal = batch['goal'][0].numpy()
        print('batch["maze"]', batch['maze'].shape) # torch.Size([1, 5, 5])
        print('batch["trajectory"]', batch['trajectory'].shape) # torch.Size([1, 10, 2])
        print('batch["start"]', batch['start'].shape) # torch.Size([1, 2])
        print('batch["goal"]', batch['goal'].shape) # torch.Size([1, 2])
        plot_maze_traj(maze, traj, start=start, goal=goal)
        break

# %%
