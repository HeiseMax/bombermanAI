from collections import namedtuple, deque
from sklearn import tree
import pickle
from typing import List

from .additional_game_states import create_additional_states
from .callbacks import choose_mode

# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action'))

# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 6400  # keep only ... last transitions


def setup_training(self):
    """
    Initialise self for training purpose.
    This is called after `setup` in callbacks.py.
    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
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
    self.logger.debug(
        f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')

    if self_action is None:
        self_action = "WAIT"

    else:
        if old_game_state is not None:
            mode = choose_mode(old_game_state)
            if True:
                states = create_additional_states(old_game_state)
                for state in states:
                    self.transitions.append(Transition(
                        self.modes[self.train_mode](state[0]), state[1][self_action]))


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

    if len(self.transitions) >= 6000:
        X = []
        Y = []
        for transition in self.transitions:
            if transition is not None:
                if transition[0] is not None:
                    X.append(transition[0])
                    Y.append(transition[1])

        self.transitions.clear()

        dtree = tree.DecisionTreeClassifier()
        dtree = dtree.fit(X, Y)

        self.model[self.train_mode].append(dtree)

        # Store the model
        with open("my-saved-model.pt", "wb") as file:
            pickle.dump(self.model, file)
