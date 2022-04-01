import os
import pickle
import random
from random import shuffle
from collections import deque
import numpy as np

from .state_to_features import state_to_features_bomb, state_to_features_crates, state_to_features_bomb_2, state_to_features_crates2, state_to_features_fight, state_to_features_default
from .rule_based_moves import rule_based_act
from .possible_actions import possible_actions

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

#MODES = {"DEFAULT": state_to_features, "DODGE_BOMB": state_to_features_bomb}

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

    self.modes = {"DEFAULT": state_to_features, "CRATES": state_to_features_crates2,
                  "DODGE_BOMB": state_to_features_bomb_2, "COIN_HEAVEN": state_to_features, "BATTLE_MODE": state_to_features_fight}
    self.train_mode = "CRATES"
    self.continue_training = False

    if self.train and not os.path.isfile("my-saved-model.pt"):
        self.logger.info("Setting up model from scratch.")
        self.model = {"DEFAULT": [], "DODGE_BOMB": []}
    else:
        self.logger.info("Loading model from saved state.")
        with open("my-saved-model.pt", "rb") as file:
            self.model = pickle.load(file)
            if self.train and not self.continue_training:
                self.model[self.train_mode] = []
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
    # Exploration vs exploitation
    random_prob = .0
    if self.train and random.random() < random_prob:
        self.logger.debug("EXPLORATION:")
        # up, right, down, left, wait, bomb
        return np.random.choice(ACTIONS, p=[.20, .20, .20, .20, .1, .1])

    # in training the rule_based_agent is playing
    if self.train:
        move = rule_based_act(self, game_state)
        return move

    # drop a bomb now and then to train bomb_dodging
    random_prob = .0
    if game_state["self"][2] and random.random() < random_prob:
        self.logger.debug("Choosing action purely at random.")
        # 80%: walk in any direction. 10% wait. 10% bomb.
        return np.random.choice(ACTIONS, p=[0, 0, 0, 0, 0, 1])

    mode = choose_mode(game_state)

    if mode == "DEFAULT":
        features = state_to_features(game_state)
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

    dforest = self.model[mode]

    predictions = []
    features = [self.modes[mode](game_state)]
    print(len(dforest))
    for dtree in dforest:
        predictions.append(dtree.predict(features))

    prob = True
    possible = True
    # prob
    if prob == True:
        actions, counts = np.unique(predictions, return_counts=True)
        if not possible:
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

    # det
    order, counts = np.unique(predictions, return_counts=True)
    return order[np.argmax(counts)]

def encode_feature(self, values):
    """
    Transform features into a single scalar used to access the q table for the current state.
    :param self: This object is passed to all callbacks and you can set arbitrary values.
    :param values: features.
    """
    #return the id of the state.
    #note that features have values starting at -1, hence the +1
    return self.state_space[values[0] + 1] [values[1] + 1] [values[2] + 1] [values[3] + 1] [values[4]] [values[5]]

def create_additional_feature_states(features):
    additional_states = []
    initial_state = features.copy()
    initial_state_rot_1 = rotate_features(initial_state)
    initial_state_rot_2 = rotate_features(initial_state_rot_1)
    initial_state_rot_3 = rotate_features(initial_state_rot_2)

    mirrored_state = mirror_features(initial_state)
    mirrored_state_rot_1 = rotate_features(mirrored_state)
    mirrored_state_rot_2 = rotate_features(mirrored_state_rot_1)
    mirrored_state_rot_3 = rotate_features(mirrored_state_rot_2)

    actions = {"RIGHT": "RIGHT", "LEFT" : "LEFT", "UP": "UP", "DOWN": "DOWN", "WAIT": "WAIT", "BOMB": "BOMB"}
    actions_rot_1 = {"RIGHT": "UP", "LEFT" : "DOWN", "UP": "LEFT", "DOWN": "RIGHT", "WAIT": "WAIT", "BOMB": "BOMB"}
    actions_rot_2 = {"RIGHT": "LEFT", "LEFT" : "RIGHT", "UP": "DOWN", "DOWN": "UP", "WAIT": "WAIT", "BOMB": "BOMB"}
    actions_rot_3 = {"RIGHT": "DOWN", "LEFT" : "UP", "UP": "RIGHT", "DOWN": "LEFT", "WAIT": "WAIT", "BOMB": "BOMB"}

    mirrored_actions = {"RIGHT": "RIGHT", "LEFT" : "LEFT", "UP": "DOWN", "DOWN": "UP", "WAIT": "WAIT", "BOMB": "BOMB"}
    mirrored_actions_rot_1 = {"RIGHT": "UP", "LEFT" : "DOWN", "UP": "RIGHT", "DOWN": "LEFT", "WAIT": "WAIT", "BOMB": "BOMB"}
    mirrored_actions_rot_2 = {"RIGHT": "LEFT", "LEFT" : "RIGHT", "UP": "UP", "DOWN": "DOWN", "WAIT": "WAIT", "BOMB": "BOMB"}
    mirrored_actions_rot_3 = {"RIGHT": "DOWN", "LEFT" : "UP", "UP": "LEFT", "DOWN": "RIGHT", "WAIT": "WAIT", "BOMB": "BOMB"}

    additional_states.append((initial_state, actions, actions))
    additional_states.append((initial_state_rot_1, actions_rot_1))
    additional_states.append((initial_state_rot_2, actions_rot_2))
    additional_states.append((initial_state_rot_3, actions_rot_3))
    additional_states.append((mirrored_state, mirrored_actions))
    additional_states.append((mirrored_state_rot_1, mirrored_actions_rot_1))
    additional_states.append((mirrored_state_rot_2, mirrored_actions_rot_2))
    additional_states.append((mirrored_state_rot_3, mirrored_actions_rot_3))

    return additional_states



def rotate_features(features):
    #rotate counter-clockwise
    temp = features.copy()
    temp[0] = features[1]
    temp[1] = features[2]
    temp[2] = features[3]
    temp[3] = features[0]
    return temp

def mirror_features(features):
    #mirror along x-axis
    temp = features.copy()
    temp[0] = features[2]
    temp[2] = features[0]
    return temp

def state_to_features(game_state: dict) -> np.array:
    """
    *This is not a required function, but an idea to structure your code.*
    Converts the game state to the input of your model, i.e.
    a feature vector.
    You can find out about the state of the game environment via game_state,
    which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
    what it contains.
    :param game_state:  A dictionary describing the current game board.
    :return: np.array
    """
    # This is the dict before the game begins and after it ends
    if game_state is None:
        return None

    #store relevant parts of the state
    field = game_state['field']
    self_pos = game_state['self'][3]
    coins = game_state['coins']
    bombs = game_state['bombs']
    bombs_loc = []
    if len(bombs) > 0:
        bombs_loc = bombs[:][0]
    ex_map = game_state['explosion_map']
    enemies_loc = []
    for i in range(len(game_state['others'])):
        enemies_loc.append(game_state['others'][i][3])

    #create map of tiles that are in radius of a bomb, and store the countdown of the respective bomb in a dictionary
    danger_map = []
    countdown = {}
    for bomb in bombs:
        bomb_pos = bomb[0]
        #store countdown of bomb because it's going to be useful later on
        countdown[bomb_pos] = bomb[1]
        if bomb_pos in danger_map:
            countdown[bomb_pos] = 10
        #add all tiles in range of the bomb and not blocked by a wall to the map
        #up
        for i in range(3):
            tile = (bomb_pos[0], bomb_pos[1] - i - 1)
            #wall stops the explosion
            if field[tile] == - 1:
                break
            elif tile not in danger_map:
                danger_map.append(tile)
                countdown[tile] = bomb[1]
            #problem: a tile can have up to 4 different countdowns if radii overlap
            #this means they have to be tracked separately. For now I just set it to 10, meaning avoid at all cost
            else:
                countdown[tile] = 10
        #right
        for i in range(3):
            tile = (bomb_pos[0] + i + 1, bomb_pos[1])
            #wall stops the explosion
            if field[tile] == - 1:
                break
            elif tile not in danger_map:
                danger_map.append(tile)
                countdown[tile] = bomb[1]
            else:
                countdown[tile] = 10
        #down
        for i in range(3):
            tile = (bomb_pos[0], bomb_pos[1] + i + 1)
            #wall stops the explosion
            if field[tile] == - 1:
                break
            elif tile not in danger_map:
                danger_map.append(tile)
                countdown[tile] = bomb[1]
            else:
                countdown[tile] = 10
        #left
        for i in range(3):
            tile = (bomb_pos[0] - i - 1, bomb_pos[1])
            #wall stops the explosion
            if field[tile] == - 1:
                break
            elif tile not in danger_map:
                danger_map.append(tile)
                countdown[tile] = bomb[1]
            else:
                countdown[tile] = 10



    #is the bomb action currently available?
    #yes = 1
    #no = 0
    if game_state['self'][2]:
        bomb_available = 1
    else:
        bomb_available = 0


    #determine how safe the currently occupied field is
    #safe = 0
    #in explosion range = 1
    #on bomb = 2
    #test with only danger as bomb information might be redundant for decision making
    self = 0
    if self_pos in danger_map:
        self = 1
    if self_pos in bombs_loc:
        self = 1


    #initialize directional features by their field values
    #crate = 1
    #empty = 0
    #wall = -1
    up = field[self_pos[0]][self_pos[1] - 1]
    right = field[self_pos[0] + 1][self_pos[1]]
    down = field[self_pos[0]][self_pos[1] + 1]
    left = field[self_pos[0] - 1][self_pos[1]]

    #if there is going to be an explosion on the tile, set feature to 4
    if (self_pos[0], self_pos[1] - 1) in danger_map and field[self_pos[0]][self_pos[1] - 1] != 1:
        up = 4
    if (self_pos[0] + 1, self_pos[1]) in danger_map and field[self_pos[0] + 1][self_pos[1]] != 1:
        right = 4
    if (self_pos[0], self_pos[1] + 1) in danger_map and field[self_pos[0]][self_pos[1] + 1] != 1:
        down = 4
    if (self_pos[0] - 1,self_pos[1]) in danger_map and field[self_pos[0] - 1][self_pos[1]] != 1:
        left = 4

    #if there is an opponent potentially blocking the tile, set feature to 4, (but only if already on it)
    if (self_pos[0], self_pos[1] - 1) in enemies_loc:
        up = 4
    if (self_pos[0] + 1, self_pos[1]) in enemies_loc:
        right = 4
    if (self_pos[0], self_pos[1] + 1) in enemies_loc:
        down = 4
    if (self_pos[0] - 1,self_pos[1]) in enemies_loc:
        left = 4

    #if there is a coin on the tile, set feature to 2, overrides danger, but can be overridden again in next step if
    #danger is immediate enough
    if (self_pos[0], self_pos[1] - 1) in coins:
        up = 2
    if (self_pos[0] + 1, self_pos[1]) in coins:
        right = 2
    if (self_pos[0], self_pos[1] + 1) in coins:
        down = 2
    if (self_pos[0] - 1,self_pos[1]) in coins:
        left = 2

    #immediate danger, meaning bomb is about to explode or just exploded, or bomb on tile
    if ((self_pos[0], self_pos[1] - 1) in bombs_loc or ex_map[self_pos[0]] [self_pos[1] - 1] > 0) and field[self_pos[0]][self_pos[1] - 1] != 1:
        up = 4
    if (self_pos[0], self_pos[1] - 1) in danger_map:
        if not 10 > countdown[(self_pos[0], self_pos[1] - 1)] > 0 and field[self_pos[0]][self_pos[1] - 1] != 1:
            up = 4
        #this is a lazy fix for a rare case where the agent places a bomb next to a coin that leads into a dead end.
        #a better version would search if the coin really leads to a dead end or not
        if self_pos in bombs_loc and field[self_pos[0]][self_pos[1] - 1] != 1:
            up = 4

    if ((self_pos[0] + 1, self_pos[1]) in bombs_loc or ex_map[self_pos[0] + 1] [self_pos[1]] > 0) and field[self_pos[0] + 1][self_pos[1]] != 1:
        right = 4
    if (self_pos[0] + 1, self_pos[1]) in danger_map:
        if not 10 > countdown[(self_pos[0] + 1, self_pos[1])] > 0 and field[self_pos[0] + 1][self_pos[1]] != 1:
            right = 4
        if self_pos in bombs_loc and field[self_pos[0] + 1][self_pos[1]] != 1:
            right = 4

    if ((self_pos[0], self_pos[1] + 1) in bombs_loc or ex_map[self_pos[0]] [self_pos[1] + 1] > 0) and field[self_pos[0]][self_pos[1] + 1] != 1:
        down = 4
    if (self_pos[0], self_pos[1] + 1) in danger_map:
        if not 10 > countdown[(self_pos[0], self_pos[1] + 1)] > 0 and field[self_pos[0]][self_pos[1] + 1] != 1:
            down = 4
        if self_pos in bombs_loc and field[self_pos[0]][self_pos[1] + 1] != 1:
            down = 4

    if ((self_pos[0] - 1, self_pos[1]) in bombs_loc or ex_map[self_pos[0] - 1] [self_pos[1]] > 0) and field[self_pos[0] - 1][self_pos[1]] != 1:
        left = 4
    if (self_pos[0] - 1, self_pos[1]) in danger_map:
        if not 10 > countdown[(self_pos[0] - 1, self_pos[1])] > 0 and field[self_pos[0] - 1][self_pos[1]] != 1:
            left = 4
        if self_pos in bombs_loc and field[self_pos[0] - 1][self_pos[1]] != 1:
            left = 4

    #for all empty tiles, calculate the one where the shortest path to the next coin starts using Dijkstra's algorithm
    if up == 0 or right == 0 or down == 0 or left == 0 or up == 4 or right == 4 or down == 4 or left == 4:
        #first, calculate the shortest path to the next coin if there is one, and the direction, one would have to go in
        coin_dist = 10000
        nearest_coin = None
        crate_score = 10000
        best_crate = None
        escape_route = None
        escape = False
        #not necessary if we already stand next to a coin
        if len(coins) != 0 and not up == 2 and not right == 2 and not down == 2 and not left == 2:
            current = self_pos
            qu = [current]
            previous = {}
            dist = {}
            dist[current] = 0
            c = 1
            success = False
            #for a real game it makes sense to set the limit a lot lower to prevent the agent from chasing after far away coins
            while c < 174 and not success:
                neighbours = [(current[0], current[1] - 1), (current[0] + 1, current[1]), (current[0], current[1] + 1), (current[0] - 1, current[1])]
                for neigh in neighbours:
                    #figure out if a field is a valid via if_statements and then store as boolean and either append to queue or find coin
                    valid = True
                    #check if neigh has already been visited
                    if neigh in qu:
                        valid = False
                    #check if there is an explosion scheduled for when we would arrive on the field
                    if neigh in danger_map:
                        if countdown[neigh] == 10 or dist[current] == countdown[neigh] + 1 or dist[current] == countdown[neigh]:
                            valid = False
                    if neigh in bombs_loc:
                        if dist[current] < countdown[neigh] + 4:
                            valid = False
                    #check if the tile is directly adjacent and there is currently an explosion that lasts for another turn
                    if ex_map[neigh] > 0 and dist[current] == 0:
                        valid = False
                    if neigh in coins and valid and neigh not in danger_map:
                        #found the shortest path coin, break out of loops and store the last node
                        success = True
                        coin_dist = dist[current]
                        break
                    if field[neigh] == 0 and valid:
                        qu.append(neigh)
                        previous[neigh] = current
                        dist[neigh] = dist[current] + 1
                #no path possible
                if not success and len(qu) <= c:
                    break
                if not success:
                    c += 1
                    current = qu[c - 1]
            #retrace the path to the neighbouring node of self
            if success:
                nearest_coin = current
                while dist[nearest_coin] > 1:
                    nearest_coin = previous[nearest_coin]

        #for searching for nearest crates it would make sense to also factor in how many crates are clustered together i.e. how many can be destroyed if a bomb was
        #placed on the empty tile. This would mean that one should search multiple crate tiles (i.e. maybe the 3 nearest)
        #and weigh them by distance and how clustered they are and return the one with the best score
        #idea: it would also be logical to favor going in a direction where there is coins and crates to blow up
        if not up == 1 and not right == 1 and not down == 1 and not left == 1 and not up == 2 and not right == 2 and not down == 2 and not left == 2:
            current = self_pos
            dist = {}
            dist[current] = 0

            qu = [current]
            previous = {}
            c = 1
            success = False
            #crate_tiles_found = 0
            #found = []
            #dist = []
            #for a real game it makes sense to set the limit a lot lower to prevent the agent from chasing after far away coins
            while c < 174 and not success:
                neighbours = [(current[0], current[1] - 1), (current[0] + 1, current[1]), (current[0], current[1] + 1), (current[0] - 1, current[1])]
                for neigh in neighbours:
                    #figure out if a field is a valid via if_statements and then store as boolean and either append to queue or find crate
                    valid = True
                    #check if neigh has already been visited
                    if neigh in qu:
                        valid = False
                    #check if there is an explosion scheduled for when we would arrive on the field
                    if neigh in danger_map:
                        if countdown[neigh] == 10 or dist[current] == countdown[neigh] + 1 or dist[current] == countdown[neigh]:
                            valid = False
                    if neigh in bombs_loc:
                        if dist[current] < countdown[neigh] + 4:
                            valid = False
                    #check if the tile is directly adjacent and there is currently an explosion that lasts for another turn
                    if ex_map[neigh] > 0 and dist[current] == 0:
                        valid = False
                    if field[neigh] == 1 and valid and current not in danger_map:
                        #if there are 3 crates adjacent and this is the first tile we find, it's the only tile we store,
                        #if it has two we search one more time etc,
                        #crate_tiles_found += 1
                        crate_score = dist[current]
                        success = True
                        break
                    #check that tile is empty and has no explosion scheduled for when the agent would step on it
                    #doesn't handle cases where multiple danger sources overlap, yet. In that case the path is just discarded.
                    if field[neigh] == 0 and valid:
                        qu.append(neigh)
                        previous[neigh] = current
                        dist[neigh] = dist[current] + 1
                if not success and len(qu) <= c:
                    break
                if not success:
                    c += 1
                    current = qu[c - 1]
            #retrace the path to the neighbouring node of self
            if success:
                """
                #calculate 'clusteredness' of each tile
                score = [1, 1, 1]
                for k in range(len(found)):
                    #up
                    for i in range(3):
                        tile = (found[k][0], found[k][1] - i - 1)
                        #wall stops the explosion
                        if field[tile] == - 1:
                            break
                        elif field[tile] == 1:
                            score[k] += 1
                    #right
                    for i in range(3):
                        tile = (found[k][0] + i + 1, found[k][1])
                        #wall stops the explosion
                        if field[tile] == - 1:
                            break
                        elif field[tile] == 1:
                            score[k] += 1
                    #down
                    for i in range(3):
                        tile = (found[k][0], found[k][1] + i + 1)
                        #wall stops the explosion
                        if field[tile] == - 1:
                            break
                        elif field[tile] == 1:
                            score[k] += 1
                    #left
                    for i in range(3):
                        tile = (found[k][0] - i - 1, found[k][1])
                        #wall stops the explosion
                        if field[tile] == - 1:
                            break
                        elif field[tile] == 1:
                            score[k] += 1
                score = np.divide(dist, score)
                best = np.argmin(score)
                c = dist[best]
                """
                best_crate = current
                while dist[best_crate] > 1:
                    best_crate = previous[best_crate]
        #get out of danger , even if there are no coins or crates or standing next to a crate or coin
        elif (nearest_coin is None and best_crate is None) and self == 1:
            current = self_pos
            qu = [current]
            previous = {}
            c = 1
            success = False
            dist = {}
            dist[current] = 0

            while c < 174 and not success:
                neighbours = [(current[0], current[1] - 1), (current[0] + 1, current[1]), (current[0], current[1] + 1), (current[0] - 1, current[1])]
                for neigh in neighbours:
                    #figure out if a field is a valid via if_statements and then store as boolean and either append to queue or find coin
                    valid = True
                    #check if neigh has already been visited
                    if neigh in qu:
                        valid = False
                    #check if there is an explosion scheduled for when we would arrive on the field
                    if neigh in danger_map:
                        if countdown[neigh] == 10 or dist[current] == countdown[neigh] + 1 or dist[current] == countdown[neigh]:
                            valid = False
                    #cant get past the bomb, if we find it, it means we are in its radius
                    if neigh in bombs_loc:
                        if dist[current] < countdown[neigh] + 4:
                            valid = False
                    #check if the tile is directly adjacent and there is currently an explosion that lasts for another turn
                    if ex_map[neigh] > 0 and dist[current] == 0:
                        valid = False
                    if field[neigh] == 0 and neigh not in danger_map and valid:
                        #found the shortest path safe field, break out of loops and store the last node
                        previous[neigh] = current
                        dist[neigh] = dist[current] + 1
                        current = neigh
                        success = True
                        break
                    if field[neigh] == 0 and valid:
                        qu.append(neigh)
                        previous[neigh] = current
                        dist[neigh] = dist[current] + 1
                #no path possible
                if not success and len(qu) <= c:
                    break
                if not success:
                    c += 1
                    current = qu[c - 1]
            #retrace the path to the neighbouring node of self
            if success:
                escape = True
                escape_route = current
                while dist[escape_route] > 1:
                    escape_route = previous[escape_route]


        if coin_dist < 10000 or crate_score < 10000 or escape:
            #hyperparameter: how to weigh coin distance
            coin_dist = coin_dist/20
            if nearest_coin is not None:
                destination = nearest_coin
            else:
                destination = best_crate
            if escape:
                destination = escape_route

            if destination == (self_pos[0], self_pos[1] - 1):
                up = 3
            if destination == (self_pos[0] + 1, self_pos[1]):
                right = 3
            if destination == (self_pos[0], self_pos[1] + 1):
                down = 3
            if destination == (self_pos[0] - 1, self_pos[1]):
                left = 3


    return [up, right, down, left, bomb_available, self]

def choose_mode(game_state):
    #return "CRATES"
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

    return "CRATES"
    return "DEFAULT"
