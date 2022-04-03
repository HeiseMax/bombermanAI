import os
import pickle
import random
from threading import currentThread

import numpy as np


ACTIONS = np.array(['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB'])


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
        # self.logger.info("Setting up model from scratch.")
        # weights = np.random.rand(38, len(ACTIONS))
        # self.model = weights / weights.sum()
        with open("my-saved-model.pt", "rb") as file:
          self.model = pickle.load(file)
    else:
        self.logger.info("Loading model from saved state.")
        with open("my-saved-model.pt", "rb") as file:
            self.model = pickle.load(file)
            #print(self.model)

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
    pr = np.array(state_to_features(game_state)@self.model)
    anp = action_not_possible(game_state)
    pr[anp] = 0
    if np.min(pr) < 0:
        pr = pr - np.minimum(pr,0)
    if np.sum(pr) == 0:
        pr = np.array([.24,.24,.24,.24,.04,.0])
        pr[anp] = 0
        pr /= np.sum(pr)
        a = np.random.choice(ACTIONS, p=pr) #ACTIONS[np.argmax(state_to_features(game_state)@self.model)]
        self.logger.debug(f'Chosen move at random: {a}')
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
