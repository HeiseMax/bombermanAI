import numpy as np

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

def possible_actions(game_state, actions=np.array(ACTIONS)):
    possible_actions = actions.copy().tolist()

    position = game_state['self'][3]
    field = game_state['field']
    bombs = game_state['bombs']
    explosion_map = game_state['explosion_map']
    self = game_state['self']
    bomb_possible = self[2]
    others = game_state['others']

    # walls and crates
    if field[position[0] - 1, position[1]] != 0:
        if "LEFT" in possible_actions:
            possible_actions.remove("LEFT")
    if field[position[0] + 1, position[1]] != 0:
        if "RIGHT" in possible_actions:
            possible_actions.remove("RIGHT")
    if field[position[0], position[1] - 1] != 0:
        if "UP" in possible_actions:
            possible_actions.remove("UP")
    if field[position[0], position[1] + 1] != 0:
        if "DOWN" in possible_actions:
            possible_actions.remove("DOWN")

    # explosions
    if explosion_map[position[0] - 1, position[1]] != 0:
        if "LEFT" in possible_actions:
            possible_actions.remove("LEFT")
    if explosion_map[position[0] + 1, position[1]] != 0:
        if "RIGHT" in possible_actions:
            possible_actions.remove("RIGHT")
    if explosion_map[position[0], position[1] - 1] != 0:
        if "UP" in possible_actions:
            possible_actions.remove("UP")
    if explosion_map[position[0], position[1] + 1] != 0:
        if "DOWN" in possible_actions:
            possible_actions.remove("DOWN")

    # bombs
    for bomb in bombs:
        if position[0] - 1 == bomb[0][0] and position[1] == bomb[0][1]:
            if "LEFT" in possible_actions:
                possible_actions.remove("LEFT")
        if position[0] + 1 == bomb[0][0] and position[1] == bomb[0][1]:
            if "RIGHT" in possible_actions:
                possible_actions.remove("RIGHT")
        if position[0] == bomb[0][0] and position[1] - 1 == bomb[0][1]:
            if "UP" in possible_actions:
                possible_actions.remove("UP")
        if position[0] == bomb[0][0] and position[1] + 1 == bomb[0][1]:
            if "DOWN" in possible_actions:
                possible_actions.remove("DOWN")

    # bomb-spaces before explosion
    deadly_spaces = []
    for bomb in bombs:
        if bomb[1] == 0:
            deadly_spaces.append(bomb[0])
            if field[bomb[0][0] - 1, bomb[0][1]] != -1:
                for i in range(1, 4):
                    deadly_spaces.append((bomb[0][0] - i, bomb[0][1]))
            if field[bomb[0][0] + 1, bomb[0][1]] != -1:
                for i in range(1, 4):
                    deadly_spaces.append((bomb[0][0] + i, bomb[0][1]))
            if field[bomb[0][0], bomb[0][1] - 1] != -1:
                for i in range(1, 4):
                    deadly_spaces.append((bomb[0][0], bomb[0][1] - i))
            if field[bomb[0][0], bomb[0][1] + 1] != -1:
                for i in range(1, 4):
                    deadly_spaces.append((bomb[0][0], bomb[0][1] + i))

    if not bomb_possible:
        if "BOMB" in possible_actions:
            possible_actions.remove("BOMB")

    for space in deadly_spaces:
        if position[0] == space[0] and position[1] == space[1]:
            if "WAIT" in possible_actions:
                possible_actions.remove("WAIT")
        if position[0] - 1 == space[0] and position[1] == space[1]:
            if "LEFT" in possible_actions:
                possible_actions.remove("LEFT")
        if position[0] + 1 == space[0] and position[1] == space[1]:
            if "RIGHT" in possible_actions:
                possible_actions.remove("RIGHT")
        if position[0] == space[0] and position[1] - 1 == space[1]:
            if "UP" in possible_actions:
                possible_actions.remove("UP")
        if position[0] == space[0] and position[1] + 1 == space[1]:
            if "DOWN" in possible_actions:
                possible_actions.remove("DOWN")

    # other players
    # for other in others:
    #     if position[0] - 1 == other[3][0] and position[1] == other[3][1]:
    #         if "LEFT" in possible_actions:
    #             possible_actions.remove("LEFT")
    #     if position[0] + 1 == other[3][0] and position[1] == other[3][1]:
    #         if "RIGHT" in possible_actions:
    #             possible_actions.remove("RIGHT")
    #     if position[0] == other[3][0] and position[1] - 1 == other[3][1]:
    #         if "UP" in possible_actions:
    #             possible_actions.remove("UP")
    #     if position[0] == other[3][0] and position[1] + 1 == other[3][1]:
    #         if "DOWN" in possible_actions:
    #             possible_actions.remove("DOWN")

    return possible_actions