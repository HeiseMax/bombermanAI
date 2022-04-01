import os
import pickle
import random
from random import shuffle
from collections import deque
import numpy as np


ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']


def setup(self):
    """Called once before a set of games to initialize data structures etc.

    The 'self' object passed to this method will be the same in all other
    callback methods. You can assign new properties (like bomb_history below)
    here or later on and they will be persistent even across multiple games.
    You can also use the self.logger object at any time to write to the log
    file for debugging (see https://docs.python.org/3.7/library/logging.html).
    """
    self.logger.debug('Successfully entered setup code')
    np.random.seed()
    # Fixed length FIFO queues to avoid repeating the same actions
    self.bomb_history = deque([], 5)
    self.coordinate_history = deque([], 20)
    # While this timer is positive, agent will not hunt/attack opponents
    self.ignore_others_timer = 0
    self.current_round = 0
    self.train_pca = False
    self.pca_features = 20

    if self.train or not os.path.isfile("my-saved-model.pt"):
        self.logger.info("Setting up model from scratch.")
        self.model = []
        with open("my-saved-pca.pt", "rb") as file:
            self.pca = pickle.load(file)
    else:
        self.logger.info("Loading model from saved state.")
        with open("my-saved-model.pt", "rb") as file:
            self.model = pickle.load(file)
        with open("my-saved-pca.pt", "rb") as file:
            self.pca = pickle.load(file)


def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    # todo Exploration vs exploitation
    random_prob = .0
    if self.train and random.random() < random_prob:
        self.logger.debug("Choosing action purely at random.")
        # 80%: walk in any direction. 10% wait. 10% bomb.
        return np.random.choice(ACTIONS, p=[.25, .25, .25, .25, 0, 0])
    #

    random_prob = .0
    if game_state["self"][2] and random.random() < random_prob and self.train:
        self.logger.debug("Choosing action purely at random.")
        # 80%: walk in any direction. 10% wait. 10% bomb.
        return np.random.choice(ACTIONS, p=[0, 0, 0, 0, 0, 1])

    if self.train:
        move = rule_based_act(self, game_state)
        return move

    print(self.pca.transform([initial_feature_flattening(game_state)]))

    dforest = self.model
    predictions = []
    #print(len(dforest))
    #print(len(initial_feature_flattening(game_state))) 586
    for dtree in dforest:
        predictions.append(dtree.predict(self.pca.transform(
            [initial_feature_flattening(game_state)])))

    prob = True
    possible = True
    # prob
    if prob == True:
        actions, counts = np.unique(predictions, return_counts=True)
        if not possible:
            p = counts / counts.sum()
            return np.random.choice(actions, p=p)
        possible_Actions = possible_actions(game_state, actions)
        p = []
        for a in possible_Actions:
            p.append(counts[np.where(actions == a)[0][0]])
        p = np.array(p)
        p = p / p.sum()
        if len(possible_Actions) == 0:
            self.logger.debug("WAIT")
            return "WAIT"
        action = np.random.choice(possible_Actions, p=p)
        self.logger.debug(possible_Actions)
        self.logger.debug(p)
        self.logger.debug(action)
        return action

    # det
    order, counts = np.unique(predictions, return_counts=True)
    return order[np.argmax(counts)]


def possible_actions(game_state, actions=np.array(ACTIONS)):
    possible_actions = actions.copy().tolist()

    position = game_state['self'][3]
    field = game_state['field']
    bombs = game_state['bombs']
    explosion_map = game_state['explosion_map']
    self = game_state['self']
    bomb_possible = self[2]
    others = game_state['others']

    # walls and crates
    if field[position[0] - 1, position[1]] != 0:
        if "LEFT" in possible_actions:
            possible_actions.remove("LEFT")
    if field[position[0] + 1, position[1]] != 0:
        if "RIGHT" in possible_actions:
            possible_actions.remove("RIGHT")
    if field[position[0], position[1] - 1] != 0:
        if "UP" in possible_actions:
            possible_actions.remove("UP")
    if field[position[0], position[1] + 1] != 0:
        if "DOWN" in possible_actions:
            possible_actions.remove("DOWN")

    # explosions
    if explosion_map[position[0] - 1, position[1]] != 0:
        if "LEFT" in possible_actions:
            possible_actions.remove("LEFT")
    if explosion_map[position[0] + 1, position[1]] != 0:
        if "RIGHT" in possible_actions:
            possible_actions.remove("RIGHT")
    if explosion_map[position[0], position[1] - 1] != 0:
        if "UP" in possible_actions:
            possible_actions.remove("UP")
    if explosion_map[position[0], position[1] + 1] != 0:
        if "DOWN" in possible_actions:
            possible_actions.remove("DOWN")

    # bombs
    for bomb in bombs:
        if position[0] - 1 == bomb[0][0] and position[1] == bomb[0][1]:
            if "LEFT" in possible_actions:
                possible_actions.remove("LEFT")
        if position[0] + 1 == bomb[0][0] and position[1] == bomb[0][1]:
            if "RIGHT" in possible_actions:
                possible_actions.remove("RIGHT")
        if position[1] == bomb[0][0] and position[1] - 1 == bomb[0][1]:
            if "UP" in possible_actions:
                possible_actions.remove("UP")
        if position[0] == bomb[0][0] and position[1] + 1 == bomb[0][1]:
            if "DOWN" in possible_actions:
                possible_actions.remove("DOWN")

    # bomb-spaces before explosion
    deadly_spaces = []
    for bomb in bombs:
        if bomb[1] == 0:
            deadly_spaces.append((bomb[0]))
            if field[bomb[0][0] - 1, bomb[0][1]] != -1:
                for i in range(1, 4):
                    deadly_spaces.append((bomb[0][0] - i, bomb[0][1]))
            if field[bomb[0][0] + 1, bomb[0][1]] != -1:
                for i in range(1, 4):
                    deadly_spaces.append((bomb[0][0] + i, bomb[0][1]))
            if field[bomb[0][0], bomb[0][1] - 1] != -1:
                for i in range(1, 4):
                    deadly_spaces.append((bomb[0][0], bomb[0][1] - i))
            if field[bomb[0][0], bomb[0][1] + 1] != -1:
                for i in range(1, 4):
                    deadly_spaces.append((bomb[0][0], bomb[0][1] + i))

    if not bomb_possible:
        if "BOMB" in possible_actions:
            possible_actions.remove("BOMB")

    for space in deadly_spaces:
        if position[0] == space[0] and position[1] == space[1]:
            if "WAIT" in possible_actions:
                possible_actions.remove("WAIT")
        if position[0] - 1 == space[0] and position[1] == space[1]:
            if "LEFT" in possible_actions:
                possible_actions.remove("LEFT")
        if position[0] + 1 == space[0] and position[1] == space[1]:
            if "RIGHT" in possible_actions:
                possible_actions.remove("RIGHT")
        if position[0] == space[0] and position[1] - 1 == space[1]:
            if "UP" in possible_actions:
                possible_actions.remove("UP")
        if position[0] == space[0] and position[1] + 1 == space[1]:
            if "DOWN" in possible_actions:
                possible_actions.remove("DOWN")

    # other players
    for other in others:
        if position[0] - 1 == other[3][0] and position[1] == other[3][1]:
            if "LEFT" in possible_actions:
                possible_actions.remove("LEFT")
        if position[0] + 1 == other[3][0] and position[1] == other[3][1]:
            if "RIGHT" in possible_actions:
                possible_actions.remove("RIGHT")
        if position[1] == other[3][0] and position[1] - 1 == other[3][1]:
            if "UP" in possible_actions:
                possible_actions.remove("UP")
        if position[0] == other[3][0] and position[1] + 1 == other[3][1]:
            if "DOWN" in possible_actions:
                possible_actions.remove("DOWN")

    return possible_actions


def create_additional_states(game_state):
    additional_states = []

    initial_game_state = game_state.copy()
    initial_game_state_rot_1 = rotate_game_state(initial_game_state).copy()
    initial_game_state_rot_2 = rotate_game_state(
        initial_game_state_rot_1).copy()
    initial_game_state_rot_3 = rotate_game_state(
        initial_game_state_rot_2).copy()

    mirrored_game_state = mirror_game_state(initial_game_state).copy()
    mirrored_game_state_rot_1 = rotate_game_state(mirrored_game_state).copy()
    mirrored_game_state_rot_2 = rotate_game_state(
        mirrored_game_state_rot_1).copy()
    mirrored_game_state_rot_3 = rotate_game_state(
        mirrored_game_state_rot_2).copy()

    actions = {"RIGHT": "RIGHT", "LEFT": "LEFT", "UP": "UP",
               "DOWN": "DOWN", "WAIT": "WAIT", "BOMB": "BOMB"}
    actions_rot_1 = actions = {"RIGHT": "UP", "LEFT": "DOWN",
                               "UP": "LEFT", "DOWN": "RIGHT", "WAIT": "WAIT", "BOMB": "BOMB"}
    actions_rot_2 = actions = {"RIGHT": "LEFT", "LEFT": "RIGHT",
                               "UP": "DOWN", "DOWN": "UP", "WAIT": "WAIT", "BOMB": "BOMB"}
    actions_rot_3 = actions = {"RIGHT": "DOWN", "LEFT": "UP",
                               "UP": "RIGHT", "DOWN": "LEFT", "WAIT": "WAIT", "BOMB": "BOMB"}

    mirrored_actions = {"RIGHT": "RIGHT", "LEFT": "LEFT",
                        "UP": "DOWN", "DOWN": "UP", "WAIT": "WAIT", "BOMB": "BOMB"}
    mirrored_actions_rot_1 = {"RIGHT": "UP", "LEFT": "DOWN",
                              "UP": "RIGHT", "DOWN": "LEFT", "WAIT": "WAIT", "BOMB": "BOMB"}
    mirrored_actions_rot_2 = {"RIGHT": "LEFT", "LEFT": "RIGHT",
                              "UP": "UP", "DOWN": "DOWN", "WAIT": "WAIT", "BOMB": "BOMB"}
    mirrored_actions_rot_3 = {"RIGHT": "DOWN", "LEFT": "UP",
                              "UP": "LEFT", "DOWN": "RIGHT", "WAIT": "WAIT", "BOMB": "BOMB"}

    additional_states.append((initial_game_state, actions))
    additional_states.append((initial_game_state_rot_1, actions_rot_1))
    additional_states.append((initial_game_state_rot_2, actions_rot_2))
    additional_states.append((initial_game_state_rot_3, actions_rot_3))
    additional_states.append((mirrored_game_state, mirrored_actions))
    additional_states.append(
        (mirrored_game_state_rot_1, mirrored_actions_rot_1))
    additional_states.append(
        (mirrored_game_state_rot_2, mirrored_actions_rot_2))
    additional_states.append(
        (mirrored_game_state_rot_3, mirrored_actions_rot_3))

    return additional_states


def rotate_coordinate_tuple(coordinate_tuple, board_size):
    new_coodinate_tuple = (
        coordinate_tuple[1], board_size - (coordinate_tuple[0] + 1))
    return new_coodinate_tuple


def flip_coordinate_tuple(coordinate_tuple, board_size):
    new_coodinate_tuple = (
        coordinate_tuple[0], board_size - (coordinate_tuple[1] + 1))
    return new_coodinate_tuple


def mirror_game_state(game_state):
    round = game_state['round']
    step = game_state['step']
    field = game_state['field']
    bombs = game_state['bombs']
    explosion_map = game_state['explosion_map']
    coins = game_state['coins']
    self = game_state['self']
    others = game_state['others']
    user_input = game_state['user_input']

    board_size = field.shape[0]

    new_field = np.flipud(field)
    new_bombs = []
    for bomb in bombs:
        new_bombs.append((flip_coordinate_tuple(bomb[0], board_size), bomb[1]))
    new_explosion_map = np.flipud(explosion_map)
    new_coins = []
    for coin in coins:
        new_coins.append(flip_coordinate_tuple(coin, board_size))
    new_self = (self[0], self[1], self[2],
                flip_coordinate_tuple(self[3], board_size))
    new_others = []
    for other in others:
        new_others.append(
            (other[0], other[1], other[2], flip_coordinate_tuple(other[3], board_size)))

    flipped_game_state = {}
    flipped_game_state['round'] = round
    flipped_game_state['step'] = step
    flipped_game_state['field'] = new_field
    flipped_game_state['bombs'] = new_bombs
    flipped_game_state['explosion_map'] = new_explosion_map
    flipped_game_state['coins'] = new_coins
    flipped_game_state['self'] = new_self
    flipped_game_state['others'] = new_others
    # this also has to be flipped to be useful
    flipped_game_state['user_input'] = user_input
    return flipped_game_state


def rotate_game_state(game_state):
    round = game_state['round']
    step = game_state['step']
    field = game_state['field']
    bombs = game_state['bombs']
    explosion_map = game_state['explosion_map']
    coins = game_state['coins']
    self = game_state['self']
    others = game_state['others']
    user_input = game_state['user_input']

    board_size = field.shape[0]

    new_field = np.rot90(field)
    new_bombs = []
    for bomb in bombs:
        new_bombs.append(
            (rotate_coordinate_tuple(bomb[0], board_size), bomb[1]))
    new_explosion_map = np.rot90(explosion_map)
    new_coins = []
    for coin in coins:
        new_coins.append(rotate_coordinate_tuple(coin, board_size))
    new_self = (self[0], self[1], self[2],
                rotate_coordinate_tuple(self[3], board_size))
    new_others = []
    for other in others:
        new_others.append(
            (other[0], other[1], other[2], rotate_coordinate_tuple(other[3], board_size)))

    rotated_game_state = {}
    rotated_game_state['round'] = round
    rotated_game_state['step'] = step
    rotated_game_state['field'] = new_field
    rotated_game_state['bombs'] = new_bombs
    rotated_game_state['explosion_map'] = new_explosion_map
    rotated_game_state['coins'] = new_coins
    rotated_game_state['self'] = new_self
    rotated_game_state['others'] = new_others
    rotated_game_state['user_input'] = user_input
    return rotated_game_state


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

    coins = game_state['coins']
    position = game_state['self'][3]
    field = game_state['field']
    bombs = game_state['bombs']
    explosions = game_state['explosion_map']

    min_xDist_coin = float("inf")
    min_yDist_coin = float("inf")
    min_dist_coin = float("inf")
    for coin in coins:
        xDist = coin[0] - position[0]
        yDist = coin[1] - position[1]
        dist = np.sqrt(xDist**2 + yDist**2)
        if dist < min_dist_coin:
            min_dist_coin = dist
            min_xDist_coin = xDist
            min_yDist_coin = yDist

    if len(coins) == 0:
        min_xDist_coin = 0
        min_yDist_coin = 0

    if len(bombs) == 0:
        min_xDist_bomb = 20
        min_yDist_bomb = 20
        bomb_timer = 400
    else:
        min_xDist_bomb = bombs[0][0][0] - position[0]
        min_yDist_bomb = bombs[0][0][1] - position[1]
        bomb_timer = bombs[0][1]

    isUpValid = field[position[1] - 1, position[0]]
    isRightValid = field[position[1], position[0] + 1]
    isDownValid = field[position[1] + 1, position[0]]
    isLeftValid = field[position[1], position[0] - 1]

    isUpExplosion = explosions[position[0], position[1] - 1]
    isRightExplosion = explosions[position[0] + 1, position[1]]
    isDownExplosion = explosions[position[0], position[1] + 1]
    isLeftExplosion = explosions[position[0] - 1, position[1]]

    channels = [min_xDist_coin, min_yDist_coin, min_xDist_bomb, min_yDist_bomb,
                bomb_timer, isUpValid, isRightValid, isDownValid, isLeftValid,
                isUpExplosion, isRightExplosion, isDownExplosion, isLeftExplosion]
    stacked_channels = np.stack(channels)
    return stacked_channels.reshape(-1)


def state_to_features_bomb(game_state: dict) -> np.array:
    if game_state is None:
        return None

    position = game_state['self'][3]
    field = game_state['field']
    bombs = game_state['bombs']
    explosions = game_state['explosion_map']
    others = game_state['others']
    board_size = field.shape[0]

    others_pos = []
    for other in range(4):
        if len(others) > other:
            others_pos.append(others[other][3])
        else:
            others_pos.append((-1, -1))

    # 25 fields around agent
    small_field = []
    for i in range(-2, 3):
        for j in range(-2, 3):
            if position[0] + i <= 0:
                small_field.append(1)
            elif position[0] + i >= board_size:
                small_field.append(1)
            elif position[1] + j <= 0:
                small_field.append(1)
            elif position[1] + j >= board_size:
                small_field.append(1)
            else:
                if np.abs(field[position[0] + i, position[1] + j]) == 1:
                    small_field.append(1)
                elif explosions[position[0] + i, position[1] + j] > 0:
                    small_field.append(1)
                elif others_pos[0][0] == position[0] + i and others_pos[0][1] == position[1] + j:
                    small_field.append(1)
                elif others_pos[1][0] == position[0] + i and others_pos[1][1] == position[1] + j:
                    small_field.append(1)
                elif others_pos[2][0] == position[0] + i and others_pos[2][1] == position[1] + j:
                    small_field.append(1)
                elif others_pos[3][0] == position[0] + i and others_pos[3][1] == position[1] + j:
                    small_field.append(1)
                else:
                    small_field.append(0)

    # 8 entries, x and y dist to all bombs
    bomb_dists = []
    for bomb in range(4):
        if len(bombs) > bomb:
            bomb_dists.append(bombs[bomb][0][0] - position[0])
            bomb_dists.append(bombs[bomb][0][1] - position[1])
        else:
            bomb_dists.append(board_size)
            bomb_dists.append(board_size)

    # 4 timers of bombs
    bomb_times = []
    for bomb in range(4):
        if len(bombs) > bomb:
            bomb_times.append(bombs[bomb][1])
        else:
            bomb_times.append(20)

    # 8 entries, x and y dist to all others
    # others_dists = []
    # for other in range(4):
    #     if len(others) > other:
    #         others_dists.append(others[other][3][0] - position[0])
    #         others_dists.append(others[other][3][1] - position[1])
    #     else:
    #         others_dists.append(board_size)
    #         others_dists.append(board_size)

    channels = []
    channels = np.append(channels, small_field)
    channels = np.append(channels, bomb_dists)
    channels = np.append(channels, bomb_times)
    #channels = np.append(channels, others_dists)
    stacked_channels = np.stack(channels)
    return stacked_channels.reshape(-1)


def initial_feature_flattening(game_state):
    round = game_state['round']
    step = game_state['step']
    field = game_state['field']
    bombs = game_state['bombs']
    explosion_map = game_state['explosion_map']
    coins = game_state['coins']
    self = game_state['self']
    others = game_state['others']

    position = self[3]

    flattened_round = [round]

    flattened_step = [step]

    flattened_field = np.array(field).flatten()

    # assumption: 1 bomb max
    if len(bombs) == 0:
        flattened_bombs = [20, 20, 400]
    else:
        flattened_bombs = [bombs[0][0][0] - position[0], bombs[0][0][1] - position[1], bombs[0][1]]

    flattened_explosion_map = np.array(explosion_map).flatten()

    # assumption: only nearest coin relevant
    min_xDist_coin = float("inf")
    min_yDist_coin = float("inf")
    min_dist_coin = float("inf")
    for coin in coins:
        xDist = coin[0] - position[0]
        yDist = coin[1] - position[1]
        dist = np.sqrt(xDist**2 + yDist**2)
        if dist < min_dist_coin:
            min_dist_coin = dist
            min_xDist_coin = xDist
            min_yDist_coin = yDist
    if len(coins) == 0:
        min_xDist_coin = 0
        min_yDist_coin = 0
    flattened_coins = [min_xDist_coin, min_yDist_coin]

    name = self[0]
    score = self[1]
    bomb_possible = self[2]
    pos_x = self[3][0]
    pos_y = self[3][1]
    flattened_self = [pos_x, pos_y, bomb_possible]  # ,score, bomb_possible]

    # assumption: no others
    flattened_others = []

    flattened_features = []
    #flattened_features = np.append(flattened_features, flattened_round)
    #flattened_features = np.append(flattened_features, flattened_step)
    flattened_features = np.append(flattened_features, flattened_field)
    flattened_features = np.append(flattened_features, flattened_bombs)
    flattened_features = np.append(flattened_features, flattened_explosion_map)
    flattened_features = np.append(flattened_features, flattened_coins)
    flattened_features = np.append(flattened_features, flattened_self)
    #flattened_features = np.append(flattened_features, flattened_others)

    return flattened_features

###################################################################################


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
    if (x - 1, y) in valid_tiles:
        valid_actions.append('LEFT')
    if (x + 1, y) in valid_tiles:
        valid_actions.append('RIGHT')
    if (x, y - 1) in valid_tiles:
        valid_actions.append('UP')
    if (x, y + 1) in valid_tiles:
        valid_actions.append('DOWN')
    if (x, y) in valid_tiles:
        valid_actions.append('WAIT')
    # Disallow the BOMB action if agent dropped a bomb in the same spot recently
    if (bombs_left > 0) and (x, y) not in self.bomb_history:
        valid_actions.append('BOMB')
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
    targets = [targets[i]
               for i in range(len(targets)) if targets[i] not in bomb_xys]

    # Take a step towards the most immediately interesting target
    free_space = arena == 0
    if self.ignore_others_timer > 0:
        for o in others:
            free_space[o] = False
    d = look_for_targets(free_space, (x, y), targets, self.logger)
    if d == (x, y - 1):
        action_ideas.append('UP')
    if d == (x, y + 1):
        action_ideas.append('DOWN')
    if d == (x - 1, y):
        action_ideas.append('LEFT')
    if d == (x + 1, y):
        action_ideas.append('RIGHT')
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
            if (yb > y):
                action_ideas.append('UP')
            if (yb < y):
                action_ideas.append('DOWN')
            # If possible, turn a corner
            action_ideas.append('LEFT')
            action_ideas.append('RIGHT')
        if (yb == y) and (abs(xb - x) < 4):
            # Run away
            if (xb > x):
                action_ideas.append('LEFT')
            if (xb < x):
                action_ideas.append('RIGHT')
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
    if len(targets) == 0:
        return None

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
        neighbors = [(x, y) for (x, y) in [(x + 1, y), (x - 1, y),
                                           (x, y + 1), (x, y - 1)] if free_space[x, y]]
        shuffle(neighbors)
        for neighbor in neighbors:
            if neighbor not in parent_dict:
                frontier.append(neighbor)
                parent_dict[neighbor] = current
                dist_so_far[neighbor] = dist_so_far[current] + 1
    if logger:
        logger.debug(f'Suitable target found at {best}')
    # Determine the first step towards the best found target tile
    current = best
    while True:
        if parent_dict[current] == start:
            return current
        current = parent_dict[current]


def reset_self(self):
    self.bomb_history = deque([], 5)
    self.coordinate_history = deque([], 20)
    # While this timer is positive, agent will not hunt/attack opponents
    self.ignore_others_timer = 0
