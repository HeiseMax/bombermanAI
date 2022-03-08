import os
import pickle
import random

import numpy as np



ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT']#, 'BOMB']


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
    if self.train or not os.path.isfile("my-saved-model.pt"):
        self.logger.info("Setting up model from scratch.")
        self.model = np.zeros((5**4,5))
    else:
        self.logger.info("Loading model from saved state.")
        with open("my-saved-model.pt", "rb") as file:
            self.model = pickle.load(file)
            #print(self.model)
    #setup k-dimensional array where k is the number of features and store an id for each possible state
    self.state_space = np.arange(0, 5**4)
    self.state_space = np.reshape(self.state_space, (5, 5, 5, 5))

def encode_feature(self, values):
    """
    Transform features into a single scalar used to access the q table for the current state.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    :param values: features.
    """
    #return the id of the state.
    #note that features have values starting at -1, hence the +1
    return self.state_space[values[0] + 1] [values[1] + 1] [values[2] + 1] [values[3] + 1]

def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    # todo Exploration vs exploitation
    random_prob = .1
    if self.train and random.random() < random_prob:
        self.logger.debug("Choosing action purely at random.")
        # 80%: walk in any direction. 10% wait. 10% bomb.
        return np.random.choice(ACTIONS, p=[.25, .25, .25, .25, .0])

    self.logger.debug("Querying model for action.")
    features = state_to_features(game_state)
    print(features)
    state = encode_feature(self, features)
    pr = [.2, .2, .2, .2, .2]
    if np.max(self.model[state]) > 0:
        pr = np.maximum(self.model[state], 0)
    pr /= np.sum(pr)
    a = np.random.choice(ACTIONS, p=pr) #ACTIONS[np.argmax(state_to_features(game_state)@self.model)]
    self.logger.debug(f'Action taken: {a}')
    return a#np.random.choice(ACTIONS, p=state_to_features(game_state)@self.model/np.sum(state_to_features(game_state)@self.model))


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
        #print('returning None')
        return None

    #store relevant parts of the state
    field = game_state['field']
    self_pos = game_state['self'][3]
    coins = game_state['coins']

    #initialize directional features by their field values
    #crate = 1
    #empty = 0
    #wall = -1
    up = field[self_pos[0]][self_pos[1] - 1]
    right = field[self_pos[0] + 1][self_pos[1]]
    down = field[self_pos[0]][self_pos[1] + 1]
    left = field[self_pos[0] - 1][self_pos[1]]

    #if there is a coin on the tile, set feature to 2
    if (self_pos[0], self_pos[1] - 1) in coins:
        up = 2
    if (self_pos[0] + 1, self_pos[1]) in coins:
        right = 2
    if (self_pos[0], self_pos[1] + 1) in coins:
        down = 2
    if (self_pos[0] - 1,self_pos[1]) in coins:
        left = 2

    #for all empty tiles, calculate the one where the shortest path to the next coin starts using Dijkstra's algorithm
    if up == 0 or right == 0 or down == 0 or left == 0 and len(coins) != 0 and not up == 2 and not right == 2 and not down == 2 and not left == 2:
        current = self_pos
        qu = [current]
        previous = [0]
        c = 0
        success = False
        destination = (0, 0)
        #for a real game it makes sense to set the limit a lot lower to prevent the agent from chasing after far away coins
        #TODO: for large numbers over 200, an error is occasionaly thrown for list index out of range at current = qu[c]
        while c < 150 and not success:
            neighbours = [(current[0], current[1] - 1), (current[0] + 1, current[1]), (current[0], current[1] + 1), (current[0] - 1, current[1])]
            for neigh in neighbours:
                if neigh in coins:
                    #found the shortest path coin, break out of loops and store the last node
                    success = True
                    destination = current
                    break
                if field[neigh] == 0 and neigh not in qu:
                    qu.append(neigh)
                    previous.append(c)
            if not success:
                c += 1
                current = qu[c]
        #retrace the path to the neighbouring node of self
        if success:
            while previous[c] != 0:
                c_temp = previous[c]
                destination = qu[previous[c]]
                c = c_temp
        if destination == (self_pos[0], self_pos[1] - 1) and success:
            up = 3
        if destination == (self_pos[0] + 1, self_pos[1]) and success:
            right = 3
        if destination == (self_pos[0], self_pos[1] + 1) and success:
            down = 3
        if destination == (self_pos[0] - 1, self_pos[1]) and success:
            left = 3


    return [up, right, down, left]
