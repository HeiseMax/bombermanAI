import numpy as np


def feature_flattening(game_state):
    round = game_state['round']
    step = game_state['step']
    field = game_state['field']
    bombs = game_state['bombs']
    explosion_map = game_state['explosion_map']
    coins = game_state['coins']
    self = game_state['self']
    others = game_state['others']

    position = self[3]

    flattened_round = [round]

    flattened_step = [step]

    flattened_field = np.array(field).flatten()

    # assumption: 1 bomb max
    if len(bombs) == 0:
        flattened_bombs = [20, 20, 400]
    else:
        flattened_bombs = [bombs[0][0][0] - position[0],
                           bombs[0][0][1] - position[1], bombs[0][1]]

    flattened_explosion_map = np.array(explosion_map).flatten()

    # assumption: only nearest coin relevant
    min_xDist_coin = float("inf")
    min_yDist_coin = float("inf")
    min_dist_coin = float("inf")
    for coin in coins:
        xDist = coin[0] - position[0]
        yDist = coin[1] - position[1]
        dist = np.sqrt(xDist**2 + yDist**2)
        if dist < min_dist_coin:
            min_dist_coin = dist
            min_xDist_coin = xDist
            min_yDist_coin = yDist
    if len(coins) == 0:
        min_xDist_coin = 0
        min_yDist_coin = 0
    flattened_coins = [min_xDist_coin, min_yDist_coin]

    name = self[0]
    score = self[1]
    bomb_possible = self[2]
    pos_x = self[3][0]
    pos_y = self[3][1]
    flattened_self = [pos_x, pos_y, bomb_possible]  # ,score, bomb_possible]

    # assumption: no others
    flattened_others = []

    flattened_features = []
    #flattened_features = np.append(flattened_features, flattened_round)
    #flattened_features = np.append(flattened_features, flattened_step)
    flattened_features = np.append(flattened_features, flattened_field)
    flattened_features = np.append(flattened_features, flattened_bombs)
    flattened_features = np.append(flattened_features, flattened_explosion_map)
    flattened_features = np.append(flattened_features, flattened_coins)
    flattened_features = np.append(flattened_features, flattened_self)
    #flattened_features = np.append(flattened_features, flattened_others)

    return flattened_features
