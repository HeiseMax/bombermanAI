import os
import pickle
import random

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
    if self.train or not os.path.isfile("my-saved-model.pt"):
        self.logger.info("Setting up model from scratch.")
        weights = np.random.rand(8, len(ACTIONS)) - 0.5
        weights[:,5] = 0
        self.model = weights / weights.sum()
        #self.model = np.array([[0.,1.,0.,0.,0.,0.],[0.,0.,0.,1.,0.,0.],[0.,0.,1.,0.,0.,0.],[1.,0.,0.,0.,0.,0.],[0.,1.,0.,0.,0.,0.],[0.,0.,0.,1.,0.,0.],[0.,0.,1.,0.,0.,0.],[1.,0.,0.,0.,0.,0.]])
        #with open("my-saved-model.pt", "rb") as file:
        #    self.model = pickle.load(file)
    else:
        self.logger.info("Loading model from saved state.")
        self.model = np.array([[0.,1.,0.,0.,0.,0.],[0.,0.,0.,1.,0.,0.],[0.,0.,1.,0.,0.,0.],[1.,0.,0.,0.,0.,0.],[0.,1.,0.,0.,0.,0.],[0.,0.,0.,1.,0.,0.],[0.,0.,1.,0.,0.,0.],[1.,0.,0.,0.,0.,0.]])
        #self.model = self.model / 100
        #print(self.model)
        #with open("my-saved-model.pt", "rb") as file:
        #    self.model = pickle.load(file)
        #    print(self.model)


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
        return np.random.choice(ACTIONS, p=[.25, .25, .25, .25, .0, .0])

    self.logger.debug("Querying model for action.")
    #pr = np.maximum(state_to_features(game_state)@self.model,1e-4)
    #pr /= np.sum(pr)
    #print(state_to_features(game_state))
    #print(self.model)
    pr = state_to_features(game_state)@self.model
    if np.min(pr) < 0:
        pr = pr - np.minimum(pr,0)
    if np.sum(pr) == 0:
        pr =[.25,.25,.25,.25,.0,.0]
    #pr /= np.sum(pr)
        a = np.random.choice(ACTIONS, p=pr) #ACTIONS[np.argmax(state_to_features(game_state)@self.model)]
    else:
        #a = ACTIONS[np.argmax(pr)]
        #pr = pr**2
        pr /= np.sum(pr)
        a = np.random.choice(ACTIONS, p=pr)
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
        return None

    # For example, you could construct several channels of equal shape, ... 
    coinLoc = game_state['coins']
    if len(coinLoc) != 0:
        selfLoc = game_state['self'][3]
        for i in range(len(coinLoc)):
            coinDistX = coinLoc[i][0] - selfLoc[0]
            coinDistY = coinLoc[i][1] - selfLoc[1]

        coinDist2Wait = coinDistX**2 + coinDistY**2
        #coinDist2PX = np.sqrt(coinDist2Wait + 2*coinDistX + 1)
        #coinDist2MX = np.sqrt(coinDist2Wait - 2*coinDistX + 1)
        #coinDist2PY = np.sqrt(coinDist2Wait + 2*coinDistY + 1)
        #coinDist2MY = np.sqrt(coinDist2Wait - 2*coinDistY + 1)
        #coinDist2Wait = np.sqrt(coinDist2Wait)
        i = np.argmin(coinDist2Wait)
        directionX = np.sign(coinLoc[i][0] - selfLoc[0])
        directionY = np.sign(coinLoc[i][1] - selfLoc[1])
    else:
        coinDist2Wait = 1
        #coinDist2PX = 0
        #coinDist2MX = 0
        #coinDist2PY = 0
        #coinDist2MY = 0
        directionX = 0
        directionY = 0
    current_pos = game_state['self'][3]
    channels = []#[1/np.min(coinDist2MY+1),1/np.min(coinDist2PX+1),1/np.min(coinDist2PY+1),1/np.min(coinDist2MX+1),1/np.min(coinDist2Wait+1)]
    if game_state['field'][current_pos[0]+1,current_pos[1]] != -1:
        channels.append(0)
    else:
        channels.append(-1)
    if game_state['field'][current_pos[0]-1,current_pos[1]] != -1:
        channels.append(0)
    else:
        channels.append(-1)
    if game_state['field'][current_pos[0],current_pos[1]+1] != -1:
        channels.append(0)
    else:
        channels.append(-1)
    if game_state['field'][current_pos[0],current_pos[1]-1] != -1:
        channels.append(0)
    else:
        channels.append(-1)
    if np.sign(directionX) == 1:
        channels.append(1)
    else:
        channels.append(0)
    if np.sign(directionX) == -1:
        channels.append(1)
    else:
        channels.append(0)
    if np.sign(directionY) == 1:
        channels.append(1)
    else:
        channels.append(0)
    if np.sign(directionY) == -1:
        channels.append(1)
    else:
        channels.append(0)
    #channels.append(np.sign(directionX))
    #channels.append(np.sign(directionY))
    #channels.append(directionX)
    #channels.append(directionY)
    #channels.append(...)
    # concatenate them as a feature tensor (they must have the same shape), ...
    stacked_channels = np.stack(channels)
    # and return them as a vector
    return stacked_channels.reshape(-1)
