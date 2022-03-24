from collections import namedtuple, deque
from re import A
import numpy as np
import pickle
from typing import List
from numpy import random

import events as e
from .callbacks import state_to_features

# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))
Buffer = namedtuple('Buffer',
                        ('state', 'action', 'time_state'))
# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 100  # keep only ... last transitions
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...

# Events
PLACEHOLDER_EVENT = "PLACEHOLDER"
APPROACHED_EVENT = 'APPROACHED'
NAPPROACHED_EVENT = 'NAPPOACHED'
VALID_MOVE_EVENT = 'VALID_MOVE'

def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # Example: Setup an array that will note transition tuples
    # (s, a, r, s')
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)
    self.buffer = deque(maxlen=TRANSITION_HISTORY_SIZE)
    self.Y_tau_t = []

def n_step_Y(self, gamma:float, current_t: int):
    Y = 0
    for i in range(len(self.transitions)):
        Y += gamma**(i)*self.transitions[i][3]
        if i + current_t == 399:                    #in case the number of recorded states is less than TRANSITION_HISTORY_SIZE
            break
    Y += gamma**TRANSITION_HISTORY_SIZE*np.max(self.transitions[0][2]@self.model)      #not 100% sure if it is not in the sum
    return Y

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
    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')


    for i in range(len(events)):
        if events[i] == e.INVALID_ACTION:
            break
        elif events[i] == e.WAITED:
            events.append(e.INVALID_ACTION)
        elif i == len(events)-1:
            events.append(VALID_MOVE_EVENT)

    #Idea: Add your own events to hand out rewards
    if old_game_state is not None and new_game_state is not None:
        #if state_to_features(old_game_state)[5] < state_to_features(new_game_state)[5] and state_to_features(old_game_state)[6]  < state_to_features(new_game_state)[6]:
        #    events.append(NAPPROACHED_EVENT)
        #app = False
        if self_action == 'UP' and state_to_features(old_game_state)[7] == 1:
            events.append(APPROACHED_EVENT)
            #app = True
        if self_action == 'DOWN' and state_to_features(old_game_state)[6] == 1:
            events.append(APPROACHED_EVENT)
            #app = True
        if self_action == 'RIGHT' and state_to_features(old_game_state)[4] == 1:
            events.append(APPROACHED_EVENT)
            #app = True
        if self_action == 'LEFT' and state_to_features(old_game_state)[5] == 1:
            events.append(APPROACHED_EVENT)
            #app = True
        #if not app:
            #events.append(NAPPROACHED_EVENT)
    # state_to_features is defined in callbacks.py
    self.transitions.append(Transition(state_to_features(old_game_state), self_action, state_to_features(new_game_state), reward_from_events(self, events)))
    p = random.uniform()
    pcheck = TRANSITION_HISTORY_SIZE/400
    #select up to 10 different transitions randomly
    if p <= pcheck  and len(self.buffer)<TRANSITION_HISTORY_SIZE and old_game_state is not None:
        self.buffer.append(Buffer(state_to_features(old_game_state), self_action, old_game_state['step']))
    if old_game_state is not None:
        if old_game_state['step']>TRANSITION_HISTORY_SIZE:
            self.Y_tau_t.append(n_step_Y(self, gamma=0.6, current_t=old_game_state['step']))


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
    for i in range(TRANSITION_HISTORY_SIZE):
        self.Y_tau_t.append(n_step_Y(self, gamma=0.6, current_t=400-i))
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')
    self.logger.debug(f'Length Y vector {len(self.Y_tau_t)}')
    learningrate = 1e-1
    model = self.model
    #self.transitions.append(Transition(state_to_features(last_game_state), last_action, None, reward_from_events(self, events)))
    for i in range(len(self.buffer)):
        action_number = 0
        actions = self.buffer[i][1]
        if actions == 'RIGHT':
            action_number = 1
        if actions == 'DOWN':
            action_number = 2
        if actions == 'LEFT':
            action_number = 3
        if actions == 'WAIT':
            action_number = 4
        self.logger.debug(f'{self.buffer[i][2]}')
        self.model[:,action_number] += learningrate/TRANSITION_HISTORY_SIZE*self.buffer[i][0]*(self.Y_tau_t[self.buffer[i][2]] - self.buffer[i][0]@model[:,action_number])
        self.logger.debug(f'Y value [{i}]: {self.Y_tau_t[self.buffer[i][2]]}')
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
        e.COIN_COLLECTED: 2,
        e.KILLED_OPPONENT: 5,
        e.INVALID_ACTION : -1,#-0.5,
        PLACEHOLDER_EVENT: -.1,  # idea: the custom event is bad
        NAPPROACHED_EVENT: 0,#-0.2,
        APPROACHED_EVENT: 0,#0.1,
        VALID_MOVE_EVENT: 0,#-0.05
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum
