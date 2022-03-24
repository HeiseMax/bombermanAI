from collections import namedtuple, deque
from os import stat
from re import A
import numpy as np
import pickle
from typing import List

from pygame import event

import events as e
from .callbacks import state_to_features

# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 1  # keep only ... last transitions
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...

# Events
PLACEHOLDER_EVENT = "PLACEHOLDER"
APPROACHED_EVENT = 'APPROACHED'
NAPPROACHED_EVENT = 'NAPPOACHED'
VALID_MOVE_EVENT = 'VALID_MOVE'
BOMB_NEXT_CRATE_EVENT = 'BOMB_NEXT_CRATE'
DANGER_EVENT = 'DANGER'
TIME_PASSED_EVENT = 'TIME_PASSED'
EVADED_EVENT = 'EVADED'

def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # Example: Setup an array that will note transition tuples
    # (s, a, r, s')
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)


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


    # for i in range(len(events)):
    #     if events[i] == e.INVALID_ACTION:
    #         break
    #     elif events[i] == 'WAIT':
    #         events.append(e.INVALID_ACTION)
    #     elif i == len(events)-1:
    #         events.append(VALID_MOVE_EVENT)

    # Idea: Add your own events to hand out rewards
    if old_game_state is not None and new_game_state is not None:
        #if state_to_features(old_game_state)[5] < state_to_features(new_game_state)[5] and state_to_features(old_game_state)[6]  < state_to_features(new_game_state)[6]:
        #    events.append(NAPPROACHED_EVENT)
        #app = False
        if len(events) != 0:
            events.append(TIME_PASSED_EVENT)
        if self_action == 'BOMB' and np.any(state_to_features(old_game_state)[8:], where = 1) and not e.INVALID_ACTION in events:
            events.append(BOMB_NEXT_CRATE_EVENT)
        # if np.any(state_to_features(old_game_state), where = 1):
        #     events.append(DANGER_EVENT)
        if  not np.all(state_to_features(old_game_state)[:4], where = 1) and np.all(state_to_features(new_game_state)[:4], where = 1):
            events.append(EVADED_EVENT)
        if self_action == 'UP' and state_to_features(old_game_state)[5] != 0:
            events.append(DANGER_EVENT)
        if self_action == 'DOWN' and state_to_features(old_game_state)[4] != 0:
            events.append(DANGER_EVENT)
        if self_action == 'RIGHT' and state_to_features(old_game_state)[6] != 0:
            events.append(DANGER_EVENT)
        if self_action == 'LEFT'  and state_to_features(old_game_state)[7] != 0:
            events.append(DANGER_EVENT)
        if self_action == 'WAIT' and not np.all(state_to_features(old_game_state)[:4], where = 1):
            events.append(DANGER_EVENT)
    self.transitions.append(Transition(state_to_features(old_game_state), self_action, state_to_features(new_game_state), reward_from_events(self, events)))

    currentModel = self.model

    for i in range(len(self.transitions)):
        update = np.zeros_like(currentModel)
        if self.transitions[i][2] is None or self.transitions[i][0] is None:
            break
        
        self_action = self.transitions[i][1]
        weightAction = currentModel[:,0]
        if self_action == 'RIGHT':
            weightAction = currentModel[:,1]
        elif self_action == 'DOWN':
            weightAction = currentModel[:,2]
        elif self_action == 'LEFT':
            weightAction = currentModel[:,3]
        elif self_action == 'WAIT':
            weightAction = currentModel[:,4]
        elif self_action == 'BOMB':
            weightAction = currentModel[:,5]
        Y = self.transitions[i][3] + 0.1*self.transitions[i][2]@weightAction
            #self.logger.debug(Y)
            #print(new_game_state['step'], i, currentModel)
            #self.logger.debug(currentModel)
            #update += self.transitions[i][0].T@(Y*np.ones((9,5)) - self.transitions[i][0]@currentModel)
        if self_action == 'RIGHT':
            update[:,1] += self.transitions[i][0] * (Y - self.transitions[i][0]@currentModel[:,1])
        if self_action == 'DOWN':
            update[:,2] += self.transitions[i][0] * (Y - self.transitions[i][0]@currentModel[:,2])
        if self_action == 'LEFT':
            update[:,3] += self.transitions[i][0] * (Y - self.transitions[i][0]@currentModel[:,3])
        if self_action == 'WAIT':
            update[:,4] += self.transitions[i][0] * (Y - self.transitions[i][0]@currentModel[:,4])
        if self_action == 'UP':
            update[:,0] += self.transitions[i][0] * (Y - self.transitions[i][0]@currentModel[:,0])
        if self_action == 'BOMB':
            update[:,5] += self.transitions[i][0] * (Y - self.transitions[i][0]@currentModel[:,5])
            
        currentModel += 0.01/TRANSITION_HISTORY_SIZE*update
        self.model = currentModel


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
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')
    self.transitions.append(Transition(state_to_features(last_game_state), last_action, None, reward_from_events(self, events)))

    currentModel = self.model
    for i in range(len(self.transitions)):
        update = np.zeros_like(currentModel)
        if self.transitions[i][2] is None or self.transitions[i][0] is None:
            break
        
        self_action = self.transitions[i][1]
        weightAction = currentModel[:,0]
        if self_action == 'RIGHT':
            weightAction = currentModel[:,1]
        elif self_action == 'DOWN':
            weightAction = currentModel[:,2]
        elif self_action == 'LEFT':
            weightAction = currentModel[:,3]
        elif self_action == 'WAIT':
            weightAction = currentModel[:,4]
        elif self_action == 'BOMB':
            weightAction = currentModel[:,5]
        Y = self.transitions[i][3] + 0.1*self.transitions[i][2]@weightAction
        self.logger.debug(Y)
            #self.logger.debug(Y)
            #print(new_game_state['step'], i, currentModel)
            #self.logger.debug(currentModel)
            #update += self.transitions[i][0].T@(Y*np.ones((9,5)) - self.transitions[i][0]@currentModel)
        if self_action == 'RIGHT':
            update[:,1] += self.transitions[i][0] * (Y - self.transitions[i][0]@currentModel[:,1])
        if self_action == 'DOWN':
            update[:,2] += self.transitions[i][0] * (Y - self.transitions[i][0]@currentModel[:,2])
        if self_action == 'LEFT':
            update[:,3] += self.transitions[i][0] * (Y - self.transitions[i][0]@currentModel[:,3])
        if self_action == 'WAIT':
            update[:,4] += self.transitions[i][0] * (Y - self.transitions[i][0]@currentModel[:,4])
        if self_action == 'UP':
            update[:,0] += self.transitions[i][0] * (Y - self.transitions[i][0]@currentModel[:,0])
        if self_action == 'BOMB':
            update[:,5] += self.transitions[i][0] * (Y - self.transitions[i][0]@currentModel[:,5])
            
        currentModel += 0.01/TRANSITION_HISTORY_SIZE*update
        self.model = currentModel

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
        e.COIN_COLLECTED: 1,
        e.KILLED_OPPONENT: 5,
        e.INVALID_ACTION : -0.1,
        e.KILLED_SELF: -1,
        e.CRATE_DESTROYED : 1,
        e.SURVIVED_ROUND: 2,
        e.GOT_KILLED: -1,
        e.BOMB_DROPPED: -0.05,
        e.WAITED : -0.01,
        PLACEHOLDER_EVENT: -.1,  # idea: the custom event is bad
        NAPPROACHED_EVENT: 0,
        APPROACHED_EVENT: 0.25,
        VALID_MOVE_EVENT: 0,
        BOMB_NEXT_CRATE_EVENT: 0.2,
        DANGER_EVENT: -0.3,
        TIME_PASSED_EVENT: 0.01,
        EVADED_EVENT: 0.3
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum
