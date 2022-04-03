import os
import pickle
import random
from random import shuffle
from collections import deque

from nbformat import current_nbformat
import settings as s

import numpy as np


ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']


def setup(self):
    """
    Setup your code. This is called once when loading each agent.
    Make sure that you prepare everything such that act(...) can be called.

    When in training mode, the separate `setup_training` in train.py is called
    after this method. This separation allows you to share your trained agent
    with other students, without revealing your training code.

    In this example, our model is a set of probabilities over actions
    that are is independent of the game state.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    self.logger.debug('Successfully entered setup code')
    np.random.seed()
    # Fixed length FIFO queues to avoid repeating the same actions
    self.bomb_history = deque([], 5)
    self.coordinate_history = deque([], 20)
    # While this timer is positive, agent will not hunt/attack opponents
    self.ignore_others_timer = 0
    self.current_round = 0
    if self.train or not os.path.isfile("my-saved-model.pt"):
        # self.logger.info("Setting up model from scratch.")
        # weights = np.random.rand(38, len(ACTIONS))
        # self.model = weights / weights.sum()
        with open("my-saved-model.pt", "rb") as file:
          self.model = pickle.load(file)
    else:
        self.logger.info("Loading model from saved state.")
        with open("my-saved-model.pt", "rb") as file:
            self.model = pickle.load(file)


def action_not_possible(game_state):
    action_not_p = []
    current_pos = game_state['self'][3]
    bombs_next = [False, False, False, False]
    for bombs in game_state['bombs']:
        bomb = bombs[0]
        bombnear = ((bomb[1] - current_pos[1])**2 + (bomb[0] - current_pos[0])**2 <= (3-bombs[1])**2)
        if bomb[1] - current_pos[1] == -1 and bombnear:
            bombs_next[0] = True
        if bomb[1] - current_pos[1] == 1 and bombnear:
            bombs_next[2] = True
        if bomb[0] - current_pos[0] == -1 and bombnear:
            bombs_next[3] = True
        if bomb[0] - current_pos[0] == 1 and bombnear:
            bombs_next[1] = True

    if game_state['field'][current_pos[0]+1,current_pos[1]] == 0 and not bombs_next[1]:
        action_not_p.append(False)
    else:
        action_not_p.append(True)
    if game_state['field'][current_pos[0]-1,current_pos[1]] == 0 and not bombs_next[3]:
        action_not_p.append(False)
    else:
        action_not_p.append(True)
    if game_state['field'][current_pos[0],current_pos[1]+1] == 0 and not bombs_next[2]:
        action_not_p.append(False)
    else:
        action_not_p.append(True)
    if game_state['field'][current_pos[0],current_pos[1]-1]  == 0 and not bombs_next[0]:
        action_not_p.append(False)
    else:
        action_not_p.append(True)
    
    
    if game_state['explosion_map'][current_pos[0]+1,current_pos[1]] != 0:
        action_not_p[0] = True
    if game_state['explosion_map'][current_pos[0]-1,current_pos[1]] != 0:
        action_not_p[1] = True
    if game_state['explosion_map'][current_pos[0],current_pos[1]+1] != 0:
        action_not_p[2] = True
    if game_state['explosion_map'][current_pos[0],current_pos[1]-1]  != 0:
        action_not_p[3] = True
    action_not_p = np.append(action_not_p, False)
    action_not_p = np.append(action_not_p, False)
    #reshuffling of entries to fit with order of actions in ACTIONS
    action_not_p = action_not_p[[3,0,2,1,4,5]]
    return action_not_p

def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    # todo Exploration vs exploitation
    random_prob = .001
    if self.train and random.random() < random_prob:
        self.logger.debug("Choosing action purely at random.")
        # 80%: walk in any direction. 10% wait. 10% bomb.
        return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .1, .1])

    self.logger.debug("Querying model for action.")
    #pr = np.maximum(state_to_features(game_state)@self.model,1e-4)
    #pr /= np.sum(pr)
    #print(state_to_features(game_state))
    pr = state_to_features(game_state)@self.model
    anp = action_not_possible(game_state)
    pr[anp] = 0
    if np.min(pr) < 0:
        pr = np.maximum(pr,0)
    if np.sum(pr) == 0:
        pr = np.array([.24,.24,.24,.24,.04,.0])
        pr[anp] = 0
        pr /= np.sum(pr)
        a = np.random.choice(ACTIONS, p=pr) 
        self.logger.debug(f"Chosen move at random: {a}")
    else:
        pr /= np.sum(pr)
        a = np.random.choice(ACTIONS, p=pr)
        self.logger.debug(f'Action taken: {a}')
    self.logger.debug(f'probabilities: {pr}')
    return a


def state_to_features(game_state: dict) -> np.array:
    """
    *This is not a required function, but an idea to structure your code.*

    Converts the game state to the input of your model, i.e.
    a feature vector.

    You can find out about the state of the game environment via game_state,
    which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
    what it contains.

    :param game_state:  A dictionary describing the current game board.
    :return: np.array
    """
    # This is the dict before the game begins and after it ends
    if game_state is None:
        return None

    # For example, you could construct several channels of equal shape, ... 
    current_pos = game_state['self'][3]
    channels = []
    for i in range(4):
        if len(game_state['bombs']) <= i:
            channels.append(1)
            channels.append(1)
            channels.append(1)
            channels.append(1)
        else:
            bombs = game_state['bombs'][0][0]
            if current_pos[0] == bombs[0] and not current_pos[0]%2 == 0 and not np.abs(current_pos[1] - bombs[1]) == 0: 
                if np.abs(current_pos[1] - bombs[1]) < 4:
                    if current_pos[1] < bombs[1]:
                        channels.append(1)
                        channels.append(1-1/(np.abs(current_pos[1] - bombs[1])+1))
                    else:
                        channels.append(1-1/(np.abs(current_pos[1] - bombs[1])+1))
                        channels.append(1)
                else: 
                    channels.append(1)
                    channels.append(1)
            else:
                if np.abs(current_pos[1] - bombs[1]) == 0:
                    channels.append(0)
                    channels.append(0)
                else:
                    channels.append(1)
                    channels.append(1)
            
            if current_pos[1] == bombs[1] and not current_pos[1]%2 == 0 and not np.abs(current_pos[0] - bombs[0]) == 0: 
                if np.abs(current_pos[0] - bombs[0]) < 4:
                    if current_pos[0] < bombs[0]:
                        channels.append(1)
                        channels.append(1-1/(np.abs(current_pos[0] - bombs[0])+1))
                    else:
                        channels.append(1-1/(np.abs(current_pos[0] - bombs[0]+1)))
                        channels.append(1)
                else: 
                    channels.append(1)
                    channels.append(1)
            else:
                if np.abs(current_pos[0] - bombs[0]) == 0:
                    channels.append(0)
                    channels.append(0)
                else:
                    channels.append(1)
                    channels.append(1)
    
    if game_state['field'][current_pos[0] +1, current_pos[1]] == 1:
        channels.append(1)
    else:
        channels.append(0)
    if game_state['field'][current_pos[0] -1, current_pos[1]] == 1:
        channels.append(1)
    else:
        channels.append(0)
    if game_state['field'][current_pos[0], current_pos[1]+1] == 1:
        channels.append(1)
    else:
        channels.append(0)
    if game_state['field'][current_pos[0], current_pos[1]-1] == 1:
        channels.append(1)
    else:
        channels.append(0)
    
    if game_state['self'][2]:
        channels.append(1)
    else:
        channels.append(-1)
    
    others = game_state['others']
    for i in range(3):
        if i >= len(others):
            channels.append(0)
            channels.append(0)
            channels.append(0)
            channels.append(0)
        else:
            other = others[i][3]
            diffX = other[0] - current_pos[0]
            diffY = other[1] - current_pos[1]
            if diffY < 0:
                channels.append(1/np.abs(diffY))
            else:
                channels.append(0)
            if diffX > 0:
                channels.append(1/np.abs(diffX))
            else:
                channels.append(0)
            if diffY > 0:
                channels.append(1/np.abs(diffY))
            else:
                channels.append(0)
            if diffX < 0:
                channels.append(1/np.abs(diffX))
            else:
                channels.append(0)


    coinLoc = game_state['coins']
    if len(coinLoc) != 0:
        coinDistX = []
        coinDistY = []
        for i in range(len(coinLoc)):
            coinDistX.append(coinLoc[i][0] - current_pos[0])
            coinDistY.append(coinLoc[i][1] - current_pos[1])
        coinDistX = np.array(coinDistX)
        coinDistY = np.array(coinDistY)

        coinDist2Wait = coinDistX**2 + coinDistY**2
        i = np.argmin(coinDist2Wait)
        directionX = np.sign(coinLoc[i][0] - current_pos[0])
        directionY = np.sign(coinLoc[i][1] - current_pos[1])
    else:
        coinDist2Wait = 1
        directionX = 0
        directionY = 0
    if np.sign(directionY) == -1:
        channels.append(1)
    else:
        channels.append(0)
    if np.sign(directionX) == 1:
        channels.append(1)
    else:
        channels.append(0)
    if np.sign(directionY) == 1:
        channels.append(1)
    else:
        channels.append(0)
    if np.sign(directionX) == -1:
        channels.append(1)
    else:
        channels.append(0)
    arena = game_state['field']
    x= current_pos[0]
    y = current_pos[1]
    if  ([arena[x + 1, y], arena[x - 1, y], arena[x, y + 1], arena[x, y - 1]].count(0) == 1):
        channels.append(1)
    else: 
        channels.append(0)
    #channels.append(...)
    # concatenate them as a feature tensor (they must have the same shape), ...
    stacked_channels = np.stack(channels)
    # and return them as a vector
    return stacked_channels.reshape(-1)


def look_for_targets(free_space, start, targets, logger=None):
    """Find direction of closest target that can be reached via free tiles.

    Performs a breadth-first search of the reachable free tiles until a target is encountered.
    If no target can be reached, the path that takes the agent closest to any target is chosen.

    Args:
        free_space: Boolean numpy array. True for free tiles and False for obstacles.
        start: the coordinate from which to begin the search.
        targets: list or array holding the coordinates of all target tiles.
        logger: optional logger object for debugging.
    Returns:
        coordinate of first step towards closest target or towards tile closest to any target.
    """
    if len(targets) == 0: return None

    frontier = [start]
    parent_dict = {start: start}
    dist_so_far = {start: 0}
    best = start
    best_dist = np.sum(np.abs(np.subtract(targets, start)), axis=1).min()

    while len(frontier) > 0:
        current = frontier.pop(0)
        # Find distance from current position to all targets, track closest
        d = np.sum(np.abs(np.subtract(targets, current)), axis=1).min()
        if d + dist_so_far[current] <= best_dist:
            best = current
            best_dist = d + dist_so_far[current]
        if d == 0:
            # Found path to a target's exact position, mission accomplished!
            best = current
            break
        # Add unexplored free neighboring tiles to the queue in a random order
        x, y = current
        neighbors = [(x, y) for (x, y) in [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)] if free_space[x, y]]
        shuffle(neighbors)
        for neighbor in neighbors:
            if neighbor not in parent_dict:
                frontier.append(neighbor)
                parent_dict[neighbor] = current
                dist_so_far[neighbor] = dist_so_far[current] + 1
    if logger: logger.debug(f'Suitable target found at {best}')
    # Determine the first step towards the best found target tile
    current = best
    while True:
        if parent_dict[current] == start: return current
        current = parent_dict[current]

def reset_self(self):
    self.bomb_history = deque([], 5)
    self.coordinate_history = deque([], 20)
    # While this timer is positive, agent will not hunt/attack opponents
    self.ignore_others_timer = 0

def rule_based_act(self, game_state):
    """
    Called each game step to determine the agent's next action.

    You can find out about the state of the game environment via game_state,
    which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
    what it contains.
    """
    self.logger.info('Picking action according to rule set')
    # Check if we are in a different round
    if game_state["round"] != self.current_round:
        reset_self(self)
        self.current_round = game_state["round"]
    # Gather information about the game state
    arena = game_state['field']
    _, score, bombs_left, (x, y) = game_state['self']
    bombs = game_state['bombs']
    bomb_xys = [xy for (xy, t) in bombs]
    others = [xy for (n, s, b, xy) in game_state['others']]
    coins = game_state['coins']
    bomb_map = np.ones(arena.shape) * 5
    for (xb, yb), t in bombs:
        for (i, j) in [(xb + h, yb) for h in range(-3, 4)] + [(xb, yb + h) for h in range(-3, 4)]:
            if (0 < i < bomb_map.shape[0]) and (0 < j < bomb_map.shape[1]):
                bomb_map[i, j] = min(bomb_map[i, j], t)

    # If agent has been in the same location three times recently, it's a loop
    if self.coordinate_history.count((x, y)) > 2:
        self.ignore_others_timer = 5
    else:
        self.ignore_others_timer -= 1
    self.coordinate_history.append((x, y))

    # Check which moves make sense at all
    directions = [(x, y), (x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
    valid_tiles, valid_actions = [], []
    for d in directions:
        if ((arena[d] == 0) and
                (game_state['explosion_map'][d] < 1) and
                (bomb_map[d] > 0) and
                (not d in others) and
                (not d in bomb_xys)):
            valid_tiles.append(d)
    if (x - 1, y) in valid_tiles: valid_actions.append('LEFT')
    if (x + 1, y) in valid_tiles: valid_actions.append('RIGHT')
    if (x, y - 1) in valid_tiles: valid_actions.append('UP')
    if (x, y + 1) in valid_tiles: valid_actions.append('DOWN')
    if (x, y) in valid_tiles: valid_actions.append('WAIT')
    # Disallow the BOMB action if agent dropped a bomb in the same spot recently
    if (bombs_left > 0) and (x, y) not in self.bomb_history: valid_actions.append('BOMB')
    self.logger.debug(f'Valid actions: {valid_actions}')

    # Collect basic action proposals in a queue
    # Later on, the last added action that is also valid will be chosen
    action_ideas = ['UP', 'DOWN', 'LEFT', 'RIGHT']
    shuffle(action_ideas)

    # Compile a list of 'targets' the agent should head towards
    cols = range(1, arena.shape[0] - 1)
    rows = range(1, arena.shape[0] - 1)
    dead_ends = [(x, y) for x in cols for y in rows if (arena[x, y] == 0)
                 and ([arena[x + 1, y], arena[x - 1, y], arena[x, y + 1], arena[x, y - 1]].count(0) == 1)]
    crates = [(x, y) for x in cols for y in rows if (arena[x, y] == 1)]
    targets = coins + dead_ends + crates
    # Add other agents as targets if in hunting mode or no crates/coins left
    if self.ignore_others_timer <= 0 or (len(crates) + len(coins) == 0):
        targets.extend(others)

    # Exclude targets that are currently occupied by a bomb
    targets = [targets[i] for i in range(len(targets)) if targets[i] not in bomb_xys]

    # Take a step towards the most immediately interesting target
    free_space = arena == 0
    if self.ignore_others_timer > 0:
        for o in others:
            free_space[o] = False
    d = look_for_targets(free_space, (x, y), targets, self.logger)
    if d == (x, y - 1): action_ideas.append('UP')
    if d == (x, y + 1): action_ideas.append('DOWN')
    if d == (x - 1, y): action_ideas.append('LEFT')
    if d == (x + 1, y): action_ideas.append('RIGHT')
    if d is None:
        self.logger.debug('All targets gone, nothing to do anymore')
        action_ideas.append('WAIT')

    # Add proposal to drop a bomb if at dead end
    if (x, y) in dead_ends:
        action_ideas.append('BOMB')
    # Add proposal to drop a bomb if touching an opponent
    if len(others) > 0:
        if (min(abs(xy[0] - x) + abs(xy[1] - y) for xy in others)) <= 1:
            action_ideas.append('BOMB')
    # Add proposal to drop a bomb if arrived at target and touching crate
    if d == (x, y) and ([arena[x + 1, y], arena[x - 1, y], arena[x, y + 1], arena[x, y - 1]].count(1) > 0):
        action_ideas.append('BOMB')

    # Add proposal to run away from any nearby bomb about to blow
    for (xb, yb), t in bombs:
        if (xb == x) and (abs(yb - y) < 4):
            # Run away
            if (yb > y): action_ideas.append('UP')
            if (yb < y): action_ideas.append('DOWN')
            # If possible, turn a corner
            action_ideas.append('LEFT')
            action_ideas.append('RIGHT')
        if (yb == y) and (abs(xb - x) < 4):
            # Run away
            if (xb > x): action_ideas.append('LEFT')
            if (xb < x): action_ideas.append('RIGHT')
            # If possible, turn a corner
            action_ideas.append('UP')
            action_ideas.append('DOWN')
    # Try random direction if directly on top of a bomb
    for (xb, yb), t in bombs:
        if xb == x and yb == y:
            action_ideas.extend(action_ideas[:4])

    # Pick last action added to the proposals list that is also valid
    while len(action_ideas) > 0:
        a = action_ideas.pop()
        if a in valid_actions:
            # Keep track of chosen action for cycle detection
            if a == 'BOMB':
                self.bomb_history.append((x, y))

            return a
