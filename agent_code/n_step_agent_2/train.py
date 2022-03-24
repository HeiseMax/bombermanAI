from codecs import BOM
from collections import namedtuple, deque
from itertools import count
from re import A
import numpy as np
import pickle
from typing import List, final
from numpy import random
from pygame import event
from sklearn.feature_selection import f_oneway
from agent_code.escape_agent_2.train import COIN_APPROACHED_EVENT, DANGER_EVENT, EVADED_EVENT, TIME_PASSED_EVENT

import events as e
from .callbacks import state_to_features

# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'new_time_state'))  #time_state = time of game_state called 'state' in the named tuple
Buffer = namedtuple('Buffer',
                        ('state', 'action', 'time_state'))
gamma = 0.5
ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 2  # keep only ... last transitions
BUFFER_HISTORY_SIZE = 240  
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...

# Events
PLACEHOLDER_EVENT = "PLACEHOLDER"
APPROACHED_EVENT = 'APPROACHED'
NAPPROACHED_EVENT = 'NAPPROACHED'
BOMB_NEAR_CRATE_EVENT = 'BOMB_NEAR_CRATE'
COIN_NOT_COLLECTED_EVENT = 'COIN_NOT_COLLECTED'
AVOIDED_DANGER_EVENT = 'AVOIDED_DANGER'
RUN_INTO_DANGER_EVENT = 'RUN_INTO_DANGER'
BOMB_LATE_GAME_EVENT = 'BOMB_LATE_GAME'
DODGED_EVENT = 'DODGED'


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

    old_st = state_to_features(old_game_state)
    new_st = state_to_features(new_game_state)

    #unfortunately the game starts with new_game_state as the first state which is needed for events, but for the
    #rest of the game old_game_state is the relevant one.
    if old_game_state is None:
        event_state = new_st
    else:
        event_state = old_st
    events = add_events(event_state, new_st, self_action, events)

    # state_to_features is defined in callbacks.py
    reward = reward_from_events(self,events)
    self.transitions.append(Transition(state_to_features(old_game_state), self_action, state_to_features(new_game_state), reward, new_game_state['step']))
    p = random.uniform()
    pcheck = BUFFER_HISTORY_SIZE/400
    self.rewards.append(reward)
    #select up to 10 different transitions randomly
    if p <= pcheck  and len(self.buffer)<BUFFER_HISTORY_SIZE and old_game_state is not None:
        self.buffer.append(Buffer(state_to_features(old_game_state), self_action, old_game_state['step']))
    #if old_game_state is not None:
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

    events = add_events(state_to_features(last_game_state), state_to_features(last_game_state), last_action, events)

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
        #self.transitions.popleft()
        #self.logger.debug(finalTime + i)
        self.Y_tau_t.append(n_step_Y(self, gamma=gamma, current_t=finalTime + i, game_ended=True))
        self.transitions.popleft()
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')
    self.logger.debug(f'Y array length: {len(self.Y_tau_t)}')
    self.logger.debug(f'Y array: {self.Y_tau_t}')
    self.logger.debug(f'rewards length: {len(self.rewards)}')
    self.logger.debug(f'rewards: {self.rewards}')
    self.logger.debug(f'Size Buffer {len(self.buffer)}')

    learningrate = 1e-1
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
        #self.logger.debug(f'Y value [{i}]: {self.Y_tau_t[self.buffer[i][2]]}')
        #self.logger.debug(f'timestep: {self.buffer[i][2]}')
    for i in range(6):
        if counts[i] == 0:
            continue
        update[:,i] *= learningrate/counts[i]
    self.model += update
    # Store the model
    with open("my-saved-model.pt", "wb") as file:
        pickle.dump(self.model, file)
    self.Y_tau_t.clear()
    self.buffer.clear()
    self.transitions.clear()
    self.rewards.clear()


def add_events(event_state, new_state, self_action, events):
    # Idea: Add your own events to hand out rewards
    priority_exists = False
    for i in range(4):
        if event_state[i] == 3:
            priority_exists = True

    if priority_exists:
        approached = False
        #check for all directions if there was a priority and if that priority was pursued
        #of course this will not be true if we collect a coin but coin collected reward just needs to be big enough
        for i in range(4):
            if event_state[i] == 3 and action_str_to_int(self_action) == i:
                approached = True
        if approached:
            events.append(APPROACHED_EVENT)
        else:
            events.append(NAPPROACHED_EVENT)

    #event that punishes agent for not collecting a coin. Otherwise the other actions still get positive reinforcement because of discount
    lazy = False
    for i in range(4):
        if event_state[i] == 2 and action_str_to_int(self_action) != i:
            lazy = True
    if lazy:
        events.append(COIN_NOT_COLLECTED_EVENT)

    #event that positively rewards agent for not stepping into danger while being in a safe area
    cautious = False
    #check if on safe tile
    if event_state[5] == 0:
        for i in range(4):
            if event_state[i] == 4 and action_str_to_int(self_action) != i:
                cautious = True
    if cautious:
        events.append(AVOIDED_DANGER_EVENT)

    suicidal = False
    #check if suicidal
    for i in range(4):
        if event_state[i] == 4 and action_str_to_int(self_action) == i:
            suicidal = True
    if suicidal:
        events.append(RUN_INTO_DANGER_EVENT)

    #reward dropping bombs late game when there is no priority or immediate danger or crate or coin
    late_game = True
    exit_available = False
    for i in range(4):
        if event_state[i] > 0 and event_state[i] != 4:
            late_game = False
        if event_state[i] == 0:
            exit_available = True
    if late_game and e.BOMB_DROPPED in events and event_state[5] == 0 and exit_available:
        events.append(BOMB_LATE_GAME_EVENT)

    #add event indicating that a bomb was dropped next to a crate:
    near_crate = False
    for i in range(4):
        if event_state[i] == 1 and e.BOMB_DROPPED in events:
            near_crate = True
    if near_crate and exit_available:
        events.append(BOMB_NEAR_CRATE_EVENT)

    #reward moving out of danger
    dodged = False
    if event_state[5] == 1:
        if new_state[5] == 0:
            dodged = True
    if dodged:
        events.append(DODGED_EVENT)

    #print(events)
    return events

def action_str_to_int(action_string):
    action = 0
    if action_string == 'RIGHT':
        action = 1
    elif action_string == 'DOWN':
        action = 2
    elif action_string == 'LEFT':
        action = 3
    elif action_string == 'WAIT':
        action = 4
    elif action_string == 'BOMB':
        action = 5
    return action

def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*
    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        e.COIN_COLLECTED: 10,
        e.KILLED_OPPONENT: 8,
        #don't drop bombs randomly
        e.BOMB_DROPPED: -5,
        #this could positively reinforce surviving without priority
        e.BOMB_EXPLODED: 0,
        e.CRATE_DESTROYED: 0,
        #probably a bad event that mostly increases variance and can reward bad bombs
        e.COIN_FOUND: 0,
        #a bit tricky, as this can quickly disincentivize placing bombs in the correct spots but is also important fot the agent to learn to survive
        e.KILLED_SELF: 0,
        e.GOT_KILLED: -5,
        e.OPPONENT_ELIMINATED: 0,
        #maybe a bit redundant?
        e.SURVIVED_ROUND: 10,
        e.INVALID_ACTION : -5,
        e.WAITED: -1,
        e.MOVED_LEFT: -1,
        e.MOVED_RIGHT: -1,
        e.MOVED_DOWN: -1,
        e.MOVED_UP: -1,
        NAPPROACHED_EVENT: -11,
        APPROACHED_EVENT: 12,
        #effective value of this is this plus bomb_dropped
        BOMB_NEAR_CRATE_EVENT: 9,
        COIN_NOT_COLLECTED_EVENT: -4,
        AVOIDED_DANGER_EVENT: 5,
        RUN_INTO_DANGER_EVENT: -5,
        BOMB_LATE_GAME_EVENT: 9,
        DODGED_EVENT: 5,
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    #print(reward_sum)
    return reward_sum
   