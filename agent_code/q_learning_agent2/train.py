from collections import namedtuple, deque
from re import A
import numpy as np
import pickle
from typing import List

import events as e
from .callbacks import state_to_features
from .callbacks import create_additional_feature_states
from .callbacks import rotate_features
from .callbacks import mirror_features


# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 1  # keep only ... last transitions
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...
LEARNING_RATE = 0.6

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

    #setup k-dimensional array where k is the number of features and store an id for each possible state
    self.state_space = np.arange(0, 6**6)
    self.state_space = np.reshape(self.state_space, (6, 6, 6, 6, 6, 6))

    #setup q table that stores a weight for each possible state and action
    #initialize with ones since positive initial values encourage exploration in the start
    self.q_table = np.ones((6**6, 6))
    self.q_table *= 10

    self.learning_rate = 0.3
    self.round = 1




def encode_feature(self, values):
    """
    Transform features into a single scalar used to access the q table for the current state.
    :param self: This object is passed to all callbacks and you can set arbitrary values.
    :param values: features.
    """
    #return the id of the state.
    #note that directional features have values starting at -1, hence the +1
    return self.state_space[values[0] + 1] [values[1] + 1] [values[2] + 1] [values[3] + 1] [values[4]] [values[5]]

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
    self.transitions.append(Transition(old_st, self_action, new_st, reward_from_events(self, events)))

    update_q_table(self)




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
    events = add_events(state_to_features(last_game_state), state_to_features(last_game_state), last_action, events)
    self.transitions.append(Transition(state_to_features(last_game_state), last_action, None, reward_from_events(self, events)))

    update_q_table(self)
    #print(events)


    #self.learning_rate = 0.99 * self.learning_rate
    self.round += 1

    #print(self.model)
    # Store the model
    with open("my-saved-model.pt", "wb") as file:
        pickle.dump(self.model, file)


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


def update_q_table(self):
    if self.transitions[0][0] is not None and self.transitions[0][2] is not None:
        #set learning and discount rates. Learning rate should ideally be decaying during training
        #learning_rate = 0.3
        discount = 0
        #update q table

        old_features = self.transitions[0][0]
        new_features = self.transitions[0][2]

        more_old_states = create_additional_feature_states(old_features)
        more_new_states = create_additional_feature_states(new_features)

        updated_states = []
        for i in range(8):
            old_st = more_old_states[i]
            new_st = more_new_states[i]

            #often the states are already symmetrical, in which case this prevents them from being updated twice with the same data
            if old_st[0] in updated_states:
                continue

            updated_states.append(old_st[0])

            old_state = encode_feature(self, old_st[0])
            new_state = encode_feature(self, new_st[0])

            action_string = self.transitions[0][1]
            #translate action according to rotation and mirroring
            action_string = old_st[1][action_string]
            action = action_str_to_int(action_string)

            #calculate optimal future value
            q_values = []
            for a in range(6):
                q_values.append(self.q_table[new_state, a])
            max_q = max(q_values)

            self.q_table[old_state, action] = self.q_table[old_state, action] + self.learning_rate * (self.transitions[0][3] + discount * max_q - self.q_table[old_state, action])

        self.model = self.q_table

    #update for last state of a game
    if self.transitions[0][2] is None:
        old_features = self.transitions[0][0]

        more_old_states = create_additional_feature_states(old_features)

        updated_states = []
        for i in range(8):
            old_st = more_old_states[i]

            if old_st[0] in updated_states:
                continue

            updated_states.append(old_st[0])

            old_state = encode_feature(self, old_st[0])

            action_string = self.transitions[0][1]
            #translate action according to rotation and mirroring
            action_string = old_st[1][action_string]
            action = action_str_to_int(action_string)
            #print(self.q_table[old_state, action])

            self.q_table[old_state, action] = self.q_table[old_state, action] + self.learning_rate * (self.transitions[0][3] - self.q_table[old_state, action])
            #print(self.q_table[old_state, action])

        self.model = self.q_table


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
