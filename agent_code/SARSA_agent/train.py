from collections import namedtuple, deque
from hashlib import new
from os import stat
from re import A
import numpy as np
import pickle
from typing import List
from sklearn.metrics import mean_squared_error

from pygame import event

import events as e
from .callbacks import state_to_features

learningrate = 1e-2
gamma = 0.05

# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))
ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 1  # keep only ... last transitions
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...

# Events
PLACEHOLDER_EVENT = "PLACEHOLDER"
APPROACHED_EVENT = 'APPROACHED'             #approached enemy
NAPPROACHED_EVENT = 'NAPPOACHED'            # went away from bomb
TIME_PASSED_EVENT = 'TIME_PASSED'           #time passed
BOMB_DEAD_END_EVENT = 'BOMB_DEAD_END'       #placed bomb in a dead end
EVADED_EVENT = 'EVADED'                     # went out of the dangerous zone of a bomb
BOMB_NEXT_CRATE_EVENT = 'BOMB_NEXT_CRATE'   #placed bomb next to a crate
DANGER_EVENT = 'DANGER'                     # agent is in danger from a bomb
COIN_APPROACHED_EVENT = 'COIN_APPROACHED'   #approached a coin
BOMB_NEXT_ENEMY_EVENT = 'BOMB_NEXT_ENEMY'   #placed bomb next to an enemy

def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # Example: Setup an array that will note transition tuples
    # (s, a, r, s')
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)
    self.Y_tau_t = []
    self.Q_state = []

def writeSquareLoss(self):
    sqrloss = [mean_squared_error(self.Y_tau_t,self.Q_state)]
    f = open("SquaredLossCoin" + str(learningrate) + str(gamma) + ".txt", "a")
    np.savetxt(f, sqrloss, newline='\t')
    f.write('\n')
    f.close()
    return

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

    # Idea: Add your own events to hand out rewards
    foldstate = state_to_features(old_game_state)
    fnewstate = state_to_features(new_game_state)
    invalid_action = e.INVALID_ACTION in events
    if old_game_state is not None:
        events.append(TIME_PASSED_EVENT)
        if foldstate[37] == 1 and self_action == 'BOMB' and not invalid_action:
            events.append(BOMB_DEAD_END_EVENT)
        if  not np.all(foldstate[:16] == 1) and np.all(fnewstate[:16] == 1) and not invalid_action:
            events.append(EVADED_EVENT)
        if self_action == 'BOMB' and np.any(foldstate[16:20] == 1) and not invalid_action:
            events.append(BOMB_NEXT_CRATE_EVENT)
        napproached = False
        danger = False
        for i in range(4):
            if self_action == 'UP' and foldstate[1+4*i] - fnewstate[1+4*i] > 0:
                danger = True
                break
            if self_action == 'UP' and foldstate[1+4*i] - fnewstate[1+4*i] < 0:
                napproached = True
                break
            if self_action == 'RIGHT' and foldstate[3+4*i] - fnewstate[3+4*i] > 0:
                danger = True
                break
            if self_action == 'RIGHT' and foldstate[3+4*i] - fnewstate[3+4*i] < 0:
                napproached = True
                break
            if self_action == 'DOWN' and foldstate[4*i] - fnewstate[4*i] > 0:
                danger = True
                break
            if self_action == 'DOWN' and foldstate[4*i] - fnewstate[4*i] < 0:
                napproached = True
                break
            if self_action == 'LEFT' and foldstate[2+4*i] - fnewstate[2+4*i] > 0:
                danger = True
                break
            if self_action == 'LEFT' and foldstate[2+4*i] - fnewstate[2+4*i] < 0:
                napproached = True
                break
        if danger:
            events.append(DANGER_EVENT)
        if napproached:
            events.append(NAPPROACHED_EVENT)
        if self_action == 'BOMB' and not invalid_action:
            self.logger.debug(foldstate[21:33])
            for i in range(3):
                if foldstate[21+4*i:25+4*i].tolist().count(1) == 1 and foldstate[21+4*i:25+4*i].tolist().count(0) == 3:
                    events.append(BOMB_NEXT_ENEMY_EVENT)
        if (self_action == 'WAIT' or self_action == 'BOMB' or invalid_action) and not np.all(foldstate[:16] == 1):
            events.append(DANGER_EVENT)
        if not napproached and not danger:
            for i in range(4):
                if self_action == ACTIONS[i] and foldstate[33+i] == 1 and not invalid_action:
                    events.append(COIN_APPROACHED_EVENT)
                    break
            for i in range(4):
                if self_action == ACTIONS[i] and foldstate[21+i] != 0:
                    events.append(APPROACHED_EVENT)
                    break


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

        Y = self.transitions[i][3] + gamma*self.transitions[i][2]@weightAction
        self.Y_tau_t.append(Y)
        self.Q_state.append(self.transitions[i][0]@weightAction)

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
            
        currentModel += learningrate/TRANSITION_HISTORY_SIZE*update
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

        Y = self.transitions[i][3] + gamma*self.transitions[i][2]@weightAction
        self.Y_tau_t.append(Y)
        self.Q_state.append(self.transitions[i][0]@weightAction)

        self.logger.debug(Y)
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
            
        currentModel += learningrate/TRANSITION_HISTORY_SIZE*update
        self.model = currentModel

    writeSquareLoss(self)
    self.Y_tau_t.clear()
    self.Q_state.clear()

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
        e.COIN_COLLECTED: 3,
        e.KILLED_OPPONENT: 5,
        e.INVALID_ACTION: -0.2,
        e.KILLED_SELF: -2,
        e.GOT_KILLED: -3,
        e.SURVIVED_ROUND: 1,
        e.BOMB_DROPPED: -0.22,
        e.COIN_FOUND: 0.3,
        e.CRATE_DESTROYED: 0.3,
        e.WAITED:-0.1,
        EVADED_EVENT: 0.6,
        BOMB_DEAD_END_EVENT: 0.5,
        BOMB_NEXT_CRATE_EVENT: 0.35,
        DANGER_EVENT: -.75,
        PLACEHOLDER_EVENT: -.1,  # idea: the custom event is bad
        NAPPROACHED_EVENT: 0.25,
        APPROACHED_EVENT: 0.1,
        TIME_PASSED_EVENT: -0.05,
        COIN_APPROACHED_EVENT: 0.35,
        BOMB_NEXT_ENEMY_EVENT: 0.7
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum
