import os
import pickle
import random
from random import shuffle
from collections import deque
import numpy as np

from .state_to_features import state_to_features_crates, state_to_features_bomb, state_to_features_fight
from .rule_based_moves import rule_based_act
from .possible_actions import possible_actions
from .q_learning import state_to_features_qlearning, encode_feature, create_additional_feature_states, rotate_features, mirror_features

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

    self.modes = {"DEFAULT": state_to_features_qlearning,
                  "DODGE_BOMB": state_to_features_bomb, "COIN_HEAVEN": state_to_features_qlearning, "BATTLE_MODE": state_to_features_fight}
    self.train_mode = "DODGE_BOMB"
    self.continue_training = False

    if self.train and not os.path.isfile("my-saved-model.pt"):
        self.logger.info("Setting up model from scratch.")
        self.model = {}
    else:
        self.logger.info("Loading model from saved state.")
        with open("my-saved-model.pt", "rb") as file:
            self.model = pickle.load(file)
            if self.train and not self.continue_training:
                self.model[self.train_mode] = []
        # the q_learning agent can be used but not trained
        with open("my-saved-model-qlearning.pt", "rb") as file:
            self.model_qlearning = pickle.load(file)
        self.state_space = np.arange(0, 6**6)
        self.state_space = np.reshape(self.state_space, (6, 6, 6, 6, 6, 6))


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
    if self.train and game_state["self"][2] and random.random() < random_prob:
        # drop bomb
        return np.random.choice(ACTIONS, p=[0, 0, 0, 0, 0, 1])

    # in training the rule_based_agent is playing
    if self.train:
        move = rule_based_act(self, game_state)
        return move

    mode = choose_mode(game_state)

    # with this the q-learning agent could be used in chosen modes
    qlearning = True
    if qlearning and mode == "DEFAULT":
        return choose_action_qlearning(self, game_state, mode)

    # prediction with decision forest
    return choose_action_dforest(self, game_state, mode)


def choose_mode(game_state):
    position = game_state['self'][3]
    field = game_state['field']
    bombs = game_state['bombs']
    coins = game_state['coins']

    for bomb in bombs:
        bomb_x_dist = bomb[0][0] - position[0]
        bomb_y_dist = bomb[0][1] - position[1]

        # bomb-radius only on y-axis
        if field[bomb[0][0] - 1, bomb[0][1]] == -1 and field[bomb[0][0] + 1, bomb[0][1]] == -1:
            if np.abs(bomb_x_dist) <= 1:
                if np.abs(bomb_y_dist) < 5:
                    return "DODGE_BOMB"

        # bomb-radius only on x-axis
        elif field[bomb[0][0], bomb[0][1] - 1] == -1 and field[bomb[0][0], bomb[0][1] + 1] == -1:
            if np.abs(bomb_y_dist) <= 1:
                if np.abs(bomb_x_dist) < 5:
                    return "DODGE_BOMB"

        # bomb-radius on x-/ and y-axis
        else:
            if np.abs(bomb_x_dist) == 1:
                if np.abs(bomb_y_dist) < 5:
                    return "DODGE_BOMB"
            elif np.abs(bomb_x_dist) <= 1:
                if np.abs(bomb_y_dist) < 4:
                    return "DODGE_BOMB"
            if np.abs(bomb_y_dist) == 1:
                if np.abs(bomb_x_dist) < 5:
                    return "DODGE_BOMB"
            elif np.abs(bomb_y_dist) <= 1:
                if np.abs(bomb_x_dist) < 4:
                    return "DODGE_BOMB"

    hasCrate = False
    if len(coins) == 0:
        for row in field:
            if hasCrate == True:
                break
            for space in row:
                if space == 1:
                    hasCrate = True
                    break
        if hasCrate == False:
            return "BATTLE_MODE"

    return "DEFAULT"


def choose_action_dforest(self, game_state, mode):
    dforest = self.model[mode]

    predictions = []
    features = [self.modes[mode](game_state)]
    for dtree in dforest:
        predictions.append(dtree.predict(features))

    use_possible_actions = True

    actions, counts = np.unique(predictions, return_counts=True)
    if not use_possible_actions:
        p = counts / counts.sum()
        return np.random.choice(actions, p=p)
    possible_Actions = possible_actions(game_state, actions)
    p = []
    actions = actions.tolist()
    for a in possible_Actions:
        p.append(counts[actions.index(a)])
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


def choose_action_qlearning(self, game_state, mode):
    features = state_to_features_qlearning(game_state)
    state = encode_feature(self, features)

    pr = [1/6, 1/6, 1/6, 1/6, 1/6, 1/6]

    if np.max(self.model_qlearning[state]) > 0:
        pr = np.maximum(self.model_qlearning[state], 0)
    pr /= np.sum(pr)

    possible_Actions = possible_actions(game_state)
    if len(possible_Actions) == 0:
        self.logger.debug("WAIT")
        return "WAIT"
    p = []
    for a in possible_Actions:
        p.append(pr[ACTIONS.index(a)])
    p = np.array(p)
    if p.sum() == 0:
        return "WAIT"
    p = p / p.sum()
    action = np.random.choice(possible_Actions, p=p)
    return action
