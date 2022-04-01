import numpy as np


def create_additional_states(game_state):
    additional_states = []

    initial_game_state = game_state.copy()
    initial_game_state_rot_1 = rotate_game_state(initial_game_state).copy()
    initial_game_state_rot_2 = rotate_game_state(
        initial_game_state_rot_1).copy()
    initial_game_state_rot_3 = rotate_game_state(
        initial_game_state_rot_2).copy()

    mirrored_game_state = mirror_game_state(initial_game_state).copy()
    mirrored_game_state_rot_1 = rotate_game_state(mirrored_game_state).copy()
    mirrored_game_state_rot_2 = rotate_game_state(
        mirrored_game_state_rot_1).copy()
    mirrored_game_state_rot_3 = rotate_game_state(
        mirrored_game_state_rot_2).copy()

    actions = {"RIGHT": "RIGHT", "LEFT": "LEFT", "UP": "UP",
               "DOWN": "DOWN", "WAIT": "WAIT", "BOMB": "BOMB"}
    actions_rot_1 = {"RIGHT": "UP", "LEFT": "DOWN",
                               "UP": "LEFT", "DOWN": "RIGHT", "WAIT": "WAIT", "BOMB": "BOMB"}
    actions_rot_2 = {"RIGHT": "LEFT", "LEFT": "RIGHT",
                               "UP": "DOWN", "DOWN": "UP", "WAIT": "WAIT", "BOMB": "BOMB"}
    actions_rot_3 = {"RIGHT": "DOWN", "LEFT": "UP",
                               "UP": "RIGHT", "DOWN": "LEFT", "WAIT": "WAIT", "BOMB": "BOMB"}

    mirrored_actions = {"RIGHT": "RIGHT", "LEFT": "LEFT",
                        "UP": "DOWN", "DOWN": "UP", "WAIT": "WAIT", "BOMB": "BOMB"}
    mirrored_actions_rot_1 = {"RIGHT": "UP", "LEFT": "DOWN",
                              "UP": "RIGHT", "DOWN": "LEFT", "WAIT": "WAIT", "BOMB": "BOMB"}
    mirrored_actions_rot_2 = {"RIGHT": "LEFT", "LEFT": "RIGHT",
                              "UP": "UP", "DOWN": "DOWN", "WAIT": "WAIT", "BOMB": "BOMB"}
    mirrored_actions_rot_3 = {"RIGHT": "DOWN", "LEFT": "UP",
                              "UP": "LEFT", "DOWN": "RIGHT", "WAIT": "WAIT", "BOMB": "BOMB"}

    additional_states.append((initial_game_state, actions))
    additional_states.append((initial_game_state_rot_1, actions_rot_1))
    additional_states.append((initial_game_state_rot_2, actions_rot_2))
    additional_states.append((initial_game_state_rot_3, actions_rot_3))
    additional_states.append((mirrored_game_state, mirrored_actions))
    additional_states.append(
        (mirrored_game_state_rot_1, mirrored_actions_rot_1))
    additional_states.append(
        (mirrored_game_state_rot_2, mirrored_actions_rot_2))
    additional_states.append(
        (mirrored_game_state_rot_3, mirrored_actions_rot_3))

    return additional_states


def rotate_coordinate_tuple(coordinate_tuple, board_size):
    new_coodinate_tuple = (
        coordinate_tuple[1], board_size - (coordinate_tuple[0] + 1))
    return new_coodinate_tuple


def flip_coordinate_tuple(coordinate_tuple, board_size):
    new_coodinate_tuple = (
        coordinate_tuple[0], board_size - (coordinate_tuple[1] + 1))
    return new_coodinate_tuple


def mirror_game_state(game_state):
    round = game_state['round']
    step = game_state['step']
    field = game_state['field']
    bombs = game_state['bombs']
    explosion_map = game_state['explosion_map']
    coins = game_state['coins']
    self = game_state['self']
    others = game_state['others']
    user_input = game_state['user_input']

    board_size = field.shape[0]

    new_field = np.flipud(field)
    new_bombs = []
    for bomb in bombs:
        new_bombs.append((flip_coordinate_tuple(bomb[0], board_size), bomb[1]))
    new_explosion_map = np.flipud(explosion_map)
    new_coins = []
    for coin in coins:
        new_coins.append(flip_coordinate_tuple(coin, board_size))
    new_self = (self[0], self[1], self[2],
                flip_coordinate_tuple(self[3], board_size))
    new_others = []
    for other in others:
        new_others.append(
            (other[0], other[1], other[2], flip_coordinate_tuple(other[3], board_size)))

    flipped_game_state = {}
    flipped_game_state['round'] = round
    flipped_game_state['step'] = step
    flipped_game_state['field'] = new_field
    flipped_game_state['bombs'] = new_bombs
    flipped_game_state['explosion_map'] = new_explosion_map
    flipped_game_state['coins'] = new_coins
    flipped_game_state['self'] = new_self
    flipped_game_state['others'] = new_others
    # this also has to be flipped to be useful
    flipped_game_state['user_input'] = user_input
    return flipped_game_state


def rotate_game_state(game_state):
    round = game_state['round']
    step = game_state['step']
    field = game_state['field']
    bombs = game_state['bombs']
    explosion_map = game_state['explosion_map']
    coins = game_state['coins']
    self = game_state['self']
    others = game_state['others']
    user_input = game_state['user_input']

    board_size = field.shape[0]

    new_field = np.rot90(field)
    new_bombs = []
    for bomb in bombs:
        new_bombs.append(
            (rotate_coordinate_tuple(bomb[0], board_size), bomb[1]))
    new_explosion_map = np.rot90(explosion_map)
    new_coins = []
    for coin in coins:
        new_coins.append(rotate_coordinate_tuple(coin, board_size))
    new_self = (self[0], self[1], self[2],
                rotate_coordinate_tuple(self[3], board_size))
    new_others = []
    for other in others:
        new_others.append(
            (other[0], other[1], other[2], rotate_coordinate_tuple(other[3], board_size)))

    rotated_game_state = {}
    rotated_game_state['round'] = round
    rotated_game_state['step'] = step
    rotated_game_state['field'] = new_field
    rotated_game_state['bombs'] = new_bombs
    rotated_game_state['explosion_map'] = new_explosion_map
    rotated_game_state['coins'] = new_coins
    rotated_game_state['self'] = new_self
    rotated_game_state['others'] = new_others
    rotated_game_state['user_input'] = user_input
    return rotated_game_state
