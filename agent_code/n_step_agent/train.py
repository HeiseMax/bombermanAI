from collections import namedtuple, deque
from itertools import count
from re import A
from turtle import shape
import numpy as np
import pickle
from typing import List, final
from numpy import random
from pygame import event
from sklearn.metrics import mean_squared_error

import events as e
from .callbacks import state_to_features

# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'new_time_state'))  #time_state = time of game_state called 'state' in the named tuple
Buffer = namedtuple('Buffer',
                        ('state', 'action', 'time_state'))

gamma = 0.5
learningrate = 1e-2

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 10  # keep only ... last transitions
BUFFER_HISTORY_SIZE = 240  
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
    self.buffer = deque(maxlen=BUFFER_HISTORY_SIZE)
    self.Y_tau_t = []
    self.rewards = []
    self.Q_value_transition = []

def squared_loss(self):
    # sqrloss = np.zeros(1)
    # for i, Q_a in enumerate(self.Q_value_transition):
    #     sqrloss += (self.Y_tau_t[i] - Q_a)**2
    # sqrloss /= len(self.Q_value_transition)
    sqrloss = [mean_squared_error(self.Y_tau_t[1:-1],self.Q_value_transition)]
    f = open("SquaredLossDefault" + str(learningrate) + str(gamma) + ".txt", "a")
    np.savetxt(f, sqrloss, newline='\t')
    f.write('\n')
    f.close()

def n_step_Y(self, gamma:float, current_t: int, game_ended = False):
    Y = 0
    n = len(self.transitions)
    #self.logger.debug(f'Y at time {current_t}:')
    for i in range(n):
        Y += gamma**(i)*self.transitions[i][3]
    if n != 0 and not game_ended:
        Y += gamma**n*np.max(self.transitions[0][2]@self.model) 
    #self.logger.debug(Y)
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
    #Idea: Add your own events to hand out rewards
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

    # calculation of the Q value for the (next) transition (in order to avoid the problem with the first None-state)
    action_number = 0
    if self_action == 'RIGHT':
        action_number = 1
    if self_action == 'DOWN':
        action_number = 2
    if self_action == 'LEFT':
        action_number = 3
    if self_action == 'WAIT':
        action_number = 4
    if self_action == 'BOMB':
        action_number = 5
    if old_game_state is not None:
        self.Q_value_transition.append(foldstate@self.model[:,action_number])

    # state_to_features is defined in callbacks.py
    reward = reward_from_events(self,events)

    self.transitions.append(Transition(state_to_features(old_game_state), self_action, state_to_features(new_game_state), reward, new_game_state['step']))

    p = random.uniform()
    pcheck = BUFFER_HISTORY_SIZE/400

    self.rewards.append(reward)

    #select up to 10 different transitions randomly
    if p <= pcheck  and len(self.buffer)<BUFFER_HISTORY_SIZE and old_game_state is not None:
        self.buffer.append(Buffer(state_to_features(old_game_state), self_action, old_game_state['step']))
    
    if len(self.transitions)==TRANSITION_HISTORY_SIZE:
        self.Y_tau_t.append(n_step_Y(self, gamma=gamma, current_t=old_game_state['step'] - TRANSITION_HISTORY_SIZE))

    if new_game_state['step'] == 400:
        self.logger.debug(f'lenght of Y: {len(self.Y_tau_t)} ')


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
    final_rewards = reward_from_events(self, events)
    self.rewards.append(final_rewards)

    lenTransitions = len(self.transitions)

    lenBuffer = len(self.buffer)

    #self.Q_value_transition.append(np.expand_dims(state_to_features(last_game_state),axis = 0)@self.model)

    self.logger.debug('End Game')

    #in case the game ends quite early and the buffer has not gained any entries yet
    if lenBuffer == 0:
        self.logger.debug('append buffer')
        for i in range(min(BUFFER_HISTORY_SIZE, lenTransitions)):
            if self.transitions[i][0] is not None:
                self.buffer.append(Buffer(self.transitions[i][0], self.transitions[i][1], self.transitions[i][4]-1))
        lenBuffer = len(self.buffer)

    self.logger.debug(self.transitions[-1][4])

    self.transitions.append(Transition(self.transitions[-1][2], last_action, state_to_features(last_game_state),final_rewards, last_game_state['step']+1))

    lenTransitions = len(self.transitions)

    finalTime = last_game_state['step'] - lenTransitions
    for i in range(lenTransitions):
        self.Y_tau_t.append(n_step_Y(self, gamma=gamma, current_t=finalTime + i, game_ended=True))
        self.transitions.popleft()
    
    #debugging information
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')
    self.logger.debug(f'Y array length: {len(self.Y_tau_t)}')
    self.logger.debug(f'Y array: {self.Y_tau_t}')
    self.logger.debug(f'rewards length: {len(self.rewards)}')
    self.logger.debug(f'rewards: {self.rewards}')
    self.logger.debug(f'Size Buffer {len(self.buffer)}')

    model = self.model
    update = np.zeros_like(self.model)
    counts = np.zeros(6)
    #self.transitions.append(Transition(state_to_features(last_game_state), last_action, None, reward_from_events(self, events)))
    for i in range(len(self.buffer)):
        #if len(self.buffer)
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
        if actions == 'BOMB':
            action_number = 5
        update[:,action_number] += self.buffer[i][0]*(self.Y_tau_t[self.buffer[i][2]] - self.buffer[i][0]@model[:,action_number])
        counts[action_number] += 1

    for i in range(6):
        if counts[i] == 0:
            continue
        update[:,i] *= learningrate/counts[i]

    self.model += update
    # Store the model
    with open("my-saved-model.pt", "wb") as file:
        pickle.dump(self.model, file)

    #write the squared loss of the whole episode into file 
    squared_loss(self)

    self.Y_tau_t.clear()
    self.buffer.clear()
    self.transitions.clear()
    self.rewards.clear()
    self.Q_value_transition.clear()


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
