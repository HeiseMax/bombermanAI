from collections import namedtuple, deque

from sklearn import tree
from sklearn import decomposition

import pickle
import numpy as np
from typing import List

import events as e
from .callbacks import state_to_features, state_to_features_bomb
from .callbacks import initial_feature_flattening
from .callbacks import create_additional_states

# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action'))  # , 'next_state', 'reward'))
pcaTransition = namedtuple('pcaTransition',
                        ('state'))  # , 'next_state', 'reward'))

# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 3200  # keep only ... last transitions
PCA_TRANSITION_HISTORY_SIZE = 60000
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...

# Events
PLACEHOLDER_EVENT = "PLACEHOLDER"


def setup_training(self):
    """
    Initialise self for training purpose.
    This is called after `setup` in callbacks.py.
    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # Example: Setup an array that will note transition tuples
    # (s, a, r, s')
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)
    self.pcatransitions = deque(maxlen=PCA_TRANSITION_HISTORY_SIZE)


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    """
    Called once per step to allow intermediate rewards based on game events.
    When this method is called, self.events will contain a list of all game
    events relevant to your agent that occurred during the previous step. Consult
    settings.py to see what events are tracked. You can hand out rewards to your
    agent based on these events and your knowledge of the (new) game state.
    This is *one* of the places where you could update your agent.
    :param self: This object is passed to all callbacks and you can set arbitrary values.
    :param old_game_state: The state that was passed to the last call of `act`.
    :param self_action: The action that you took.
    :param new_game_state: The state the agent is in now.
    :param events: The events that occurred when going from  `old_game_state` to `new_game_state`
    """
    self.logger.debug(
        f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')


    if self_action is None:
        self_action = "WAIT"
    if self.train_pca:
        self.pcatransitions.append(pcaTransition(old_game_state))
    else:
        if old_game_state is not None:
            states = create_additional_states(old_game_state)
            for state in states:
                self.transitions.append(Transition(
                    self.pca.transform([initial_feature_flattening(state[0])]), state[1][self_action]))


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.
    This replaces game_events_occurred in this round.
    This is similar to game_events_occurred. self.events will contain all events that
    occurred during your agent's final step.
    This is *one* of the places where you could update your agent.
    This is also a good place to store an agent that you updated.
    :param self: The same object that is passed to all of your callbacks.
    """
    self.logger.debug(
        f'Encountered event(s) {", ".join(map(repr, events))} in final step')

    if self.train_pca:
        print(len(self.pcatransitions))
        if len(self.pcatransitions) >= 59000:
            X = []
            #Y = []
            for i, transition in enumerate(self.pcatransitions):
                if transition is not None:
                    if transition[0] is not None:
                        X.append(initial_feature_flattening(transition[0]))
                        # Y.append(transition[1])

            X_flattened = []
            i = 0
            for game_state in X:
                i += 1
                X_flattened.append(game_state)
            pca = decomposition.PCA(n_components=self.pca_features)
            pca.fit(X_flattened)
            self.pca = pca

            # Store the model
            with open("my-saved-pca.pt", "wb") as file:
                pickle.dump(self.pca, file)
            exit(1)
    else:
        X = []
        Y = []
        for i, transition in enumerate(self.transitions):
            if transition is not None:
                if transition[0] is not None:
                    X.append(transition[0][0])
                    Y.append(transition[1])

        dtree = tree.DecisionTreeClassifier()
        dtree = dtree.fit(X, Y)
        self.model.append(dtree)

        # Store the model
        with open("my-saved-model.pt", "wb") as file:
            pickle.dump(self.model, file)


def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*
    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        e.COIN_COLLECTED: 10,
        e.MOVED_DOWN: -0.5,
        e.MOVED_LEFT: -0.5,
        e.MOVED_RIGHT: -0.5,
        e.MOVED_UP: -0.5,
        e.WAITED: -0.5,
        e.INVALID_ACTION: -5,
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum
