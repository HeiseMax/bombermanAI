import os
import pickle
import random
from random import shuffle
from collections import deque
import numpy as np

from .feature_flattening import feature_flattening
from .rule_based_moves import rule_based_act
from .possible_actions import possible_actions


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
    # drop a bomb now and then to train bomb_dodging
    random_prob = .0
    if game_state["self"][2] and random.random() < random_prob and self.train:
        self.logger.debug("Choosing action purely at random.")
        # drop bomb
        return np.random.choice(ACTIONS, p=[0, 0, 0, 0, 0, 1])

    if self.train:
        move = rule_based_act(self, game_state)
        return move

    # print(self.pca.transform([feature_flattening(game_state)]))

    dforest = self.model
    predictions = []

    for dtree in dforest:
        predictions.append(dtree.predict(self.pca.transform(
            [feature_flattening(game_state)])))

    use_possible_actions = True

    actions, counts = np.unique(predictions, return_counts=True)
    if not use_possible_actions:
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
