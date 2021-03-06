import numpy as np

from .possible_actions import possible_actions


def state_to_features_coin_heaven(game_state: dict) -> np.array:
    if game_state is None:
        return None

    coins = game_state['coins']
    position = game_state['self'][3]
    field = game_state['field']
    bombs = game_state['bombs']
    explosions = game_state['explosion_map']

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

    channels = [min_xDist_coin, min_yDist_coin]
    stacked_channels = np.stack(channels)
    return stacked_channels.reshape(-1)


def state_to_features_crates(game_state: dict) -> np.array:
    if game_state is None:
        return None

    position = game_state['self'][3]
    coins = game_state['coins']
    field = game_state['field']
    bombs = game_state['bombs']
    explosions = game_state['explosion_map']
    others = game_state['others']
    board_size = field.shape[0]

    others_pos = []
    for other in range(4):
        if len(others) > other:
            others_pos.append(others[other][3])
        else:
            others_pos.append((-1, -1))

    # 25 fields around agent
    small_field = []
    for i in range(-2, 3):
        for j in range(-2, 3):
            if position[0] + i <= 0:
                small_field.append(1)
            elif position[0] + i >= board_size:
                small_field.append(1)
            elif position[1] + j <= 0:
                small_field.append(1)
            elif position[1] + j >= board_size:
                small_field.append(1)
            else:
                if np.abs(field[position[0] + i, position[1] + j]) == 1:
                    small_field.append(1)
                elif explosions[position[0] + i, position[1] + j] > 0:
                    small_field.append(1)
                elif others_pos[0][0] == position[0] + i and others_pos[0][1] == position[1] + j:
                    small_field.append(1)
                elif others_pos[1][0] == position[0] + i and others_pos[1][1] == position[1] + j:
                    small_field.append(1)
                elif others_pos[2][0] == position[0] + i and others_pos[2][1] == position[1] + j:
                    small_field.append(1)
                elif others_pos[3][0] == position[0] + i and others_pos[3][1] == position[1] + j:
                    small_field.append(1)
                else:
                    small_field.append(0)

    # 8 entries, x and y dist to all bombs
    bomb_dists = []
    for bomb in range(1):
        if len(bombs) > bomb:
            bomb_dists.append(bombs[bomb][0][0] - position[0])
            bomb_dists.append(bombs[bomb][0][1] - position[1])
        else:
            bomb_dists.append(board_size)
            bomb_dists.append(board_size)

    # 4 timers of bombs
    bomb_times = []
    for bomb in range(1):
        if len(bombs) > bomb:
            bomb_times.append(bombs[bomb][1])
        else:
            bomb_times.append(20)

    # 8 entries, x and y dist to all others
    # others_dists = []
    # for other in range(4):
    #     if len(others) > other:
    #         others_dists.append(others[other][3][0] - position[0])
    #         others_dists.append(others[other][3][1] - position[1])
    #     else:
    #         others_dists.append(board_size)
    #         others_dists.append(board_size)

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

    # 4 nearest crates
    crates = field == 1
    crates_pos = np.where(crates == True)
    # nearest_crates_dist = np.array([22, 22, 22, 22])
    # nearest_crates_Xdist = np.array([22, 22, 22, 22])
    # nearest_crates_Ydist = np.array([22, 22, 22, 22])

    crates_pos_x_dist = crates_pos[0] - position[0]
    crates_pos_y_dist = crates_pos[1] - position[1]

    crates_pos_x_dist_squared = np.square(crates_pos_x_dist)
    crates_pos_y_dist_squared = np.square(crates_pos_y_dist)

    crates_pos_dist_squared = crates_pos_x_dist_squared + crates_pos_y_dist_squared

    k = min(3, crates_pos_dist_squared.shape[0] - 1)
    indeces_squared = np.argpartition(crates_pos_dist_squared, k)[0:k]
    nearest_crates_Xdist = crates_pos_x_dist[indeces_squared]
    nearest_crates_Ydist = crates_pos_y_dist[indeces_squared]

    while nearest_crates_Xdist.shape[0] < 3:
        nearest_crates_Xdist = np.append(nearest_crates_Xdist, 25)
        nearest_crates_Ydist = np.append(nearest_crates_Ydist, 25)

    crates_in_radius = 0
    up_wall = False
    right_wall = False
    down_wall = False
    left_wall = False
    for i in range(3):
        if not up_wall:
            if not field[position[0], position[1] - i] == -1:
                if field[position[0], position[1] - i] == 1:
                    crates_in_radius += 1
            else:
                up_wall = True
        else:
            up_wall = True
        if not right_wall:
            if not field[position[0] + i, position[1]] == -1:
                if field[position[0] + i, position[1]] == 1:
                    crates_in_radius += 1
            else:
                right_wall = True
        else:
            right_wall = True
        if not down_wall:
            if not field[position[0], position[1] + i] == -1:
                if field[position[0], position[1] + i] == 1:
                    crates_in_radius += 1
            else:
                down_wall = True
        else:
            down_wall = True
        if not left_wall:
            if not field[position[0] - i, position[1]] == -1:
                if field[position[0] - i, position[1]] == 1:
                    crates_in_radius += 1
            else:
                left_wall = True
        else:
            left_wall = True

    channels = []
    channels = np.append(channels, min_xDist_coin)
    channels = np.append(channels, min_yDist_coin)
    channels = np.append(channels, nearest_crates_Xdist)
    channels = np.append(channels, nearest_crates_Ydist)
    channels = np.append(channels, game_state['self'][2])
    #print(crates_in_radius)
    channels = np.append(channels, crates_in_radius)

    #channels = np.append(channels, small_field)
    #channels = np.append(channels, bomb_dists)
    #channels = np.append(channels, bomb_times)
    #channels = np.append(channels, others_dists)
    stacked_channels = np.stack(channels)
    return stacked_channels.reshape(-1)



def state_to_features_fight(game_state: dict) -> np.array:
    if game_state is None:
        return None

    position = game_state['self'][3]
    field = game_state['field']
    bombs = game_state['bombs']
    explosions = game_state['explosion_map']
    others = game_state['others']
    board_size = field.shape[0]

    # 6 entries, x and y dist to all others
    others_dists = []
    for other in range(2):  # only two at the moment!! (3  max)
        if len(others) > other:
            others_dists.append(1/(others[other][3][0] - position[0] + 0.1))
            others_dists.append(1/(others[other][3][1] - position[1] + 0.1))
        else:
            others_dists.append(1/board_size)
            others_dists.append(1/board_size)

    # 8 entries, x and y dist to all bombs
    bomb_dists = []
    for bomb in range(4):
        if len(bombs) > bomb:
            bomb_dists.append(bombs[bomb][0][0] - position[0])
            bomb_dists.append(bombs[bomb][0][1] - position[1])
        else:
            bomb_dists.append(board_size)
            bomb_dists.append(board_size)

    # 4 timers of bombs
    bomb_times = []
    for bomb in range(4):
        if len(bombs) > bomb:
            bomb_times.append(bombs[bomb][1])
        else:
            bomb_times.append(20)

    min_xDist_other = float("inf")
    min_yDist_other = float("inf")
    min_dist_other = float("inf")
    for other in others:
        xDist = other[3][0] - position[0]
        yDist = other[3][1] - position[1]
        dist = np.sqrt(xDist**2 + yDist**2)
        if dist < min_dist_other:
            min_dist_other = dist
            min_xDist_other = xDist
            min_yDist_other = yDist

    if len(others) == 0:
        min_xDist_other = 17
        min_yDist_other = 17

    channels = []
    #channels = np.append(channels, [min_xDist_other, min_yDist_other, game_state['self'][2]])
    channels = np.append(channels, bomb_dists)
    channels = np.append(channels, bomb_times)
    # remove bombs, add flag, dist not 1/x again?
    channels = np.append(channels, others_dists)
    stacked_channels = np.stack(channels)
    return stacked_channels.reshape(-1)


def reachable_spaces_moves(game_state, position, steps):
    field = game_state['field']
    bombs = game_state['bombs']
    board_size = field.shape[0]

    bomb_pos_tuples = []
    for bomb in bombs:
        bomb_pos_tuples.append((bomb[0][0], bomb[0][1]))

    actions = []
    fields = []
    if steps == 0:
        return [[]], [position]

    if field[position[0], position[1] - 1] == 0 and (position[0], position[1] - 1) not in bomb_pos_tuples:
        moves_sets, fields_ret = reachable_spaces_moves(
            game_state, (position[0], position[1] - 1), steps - 1)
        fields.extend(fields_ret)
        for moves in moves_sets:
            moves.append("UP")
            actions.append(moves)
    if field[position[0] + 1, position[1]] == 0 and (position[0] + 1, position[1]) not in bomb_pos_tuples:
        moves_sets, fields_ret = reachable_spaces_moves(
            game_state, (position[0] + 1, position[1]), steps - 1)
        fields.extend(fields_ret)
        for moves in moves_sets:
            moves.append("RIGHT")
            actions.append(moves)
    if field[position[0], position[1] + 1] == 0 and (position[0], position[1] + 1) not in bomb_pos_tuples:
        moves_sets, fields_ret = reachable_spaces_moves(
            game_state, (position[0], position[1] + 1), steps - 1)
        fields.extend(fields_ret)
        for moves in moves_sets:
            moves.append("DOWN")
            actions.append(moves)
    if field[position[0] - 1, position[1]] == 0 and (position[0] - 1, position[1]) not in bomb_pos_tuples:
        moves_sets, fields_ret = reachable_spaces_moves(
            game_state, (position[0] - 1, position[1]), steps - 1)
        fields.extend(fields_ret)
        for moves in moves_sets:
            moves.append("LEFT")
            actions.append(moves)
    moves_sets, fields_ret = reachable_spaces_moves(
        game_state, (position[0], position[1]), steps - 1)
    fields.extend(fields_ret)
    for moves in moves_sets:
        moves.append("WAIT")  # maybe add BOMB?
        actions.append(moves)

    return actions, fields


def state_to_features_bomb(game_state: dict) -> np.array:
    if game_state is None:
        return None

    position = game_state['self'][3]
    field = game_state['field']
    bombs = game_state['bombs']
    explosions = game_state['explosion_map']
    others = game_state['others']
    board_size = field.shape[0]

    bomb_pos_tuples = []
    for bomb in bombs:
        bomb_pos_tuples.append((bomb[0][0], bomb[0][1]))

    others_pos_tuples = []
    for other in others:
        others_pos_tuples.append((other[3][0], other[3][1]))

    explosion_fields_in1 = []
    explosion_fields_in2 = []
    explosion_fields_in3 = []
    explosion_fields_in4 = []
    for bomb in bombs:
        if bomb[1] == 0:
            explosion_fields_in1.append(bomb[0])
            if field[bomb[0][0] - 1, bomb[0][1]] != -1:
                for i in range(1, 4):
                    explosion_fields_in1.append((bomb[0][0] - i, bomb[0][1]))
            if field[bomb[0][0] + 1, bomb[0][1]] != -1:
                for i in range(1, 4):
                    explosion_fields_in1.append((bomb[0][0] + i, bomb[0][1]))
            if field[bomb[0][0], bomb[0][1] - 1] != -1:
                for i in range(1, 4):
                    explosion_fields_in1.append((bomb[0][0], bomb[0][1] - i))
            if field[bomb[0][0], bomb[0][1] + 1] != -1:
                for i in range(1, 4):
                    explosion_fields_in1.append((bomb[0][0], bomb[0][1] + i))
        elif bomb[1] == 1:
            explosion_fields_in2.append(bomb[0])
            if field[bomb[0][0] - 1, bomb[0][1]] != -1:
                for i in range(1, 4):
                    explosion_fields_in2.append((bomb[0][0] - i, bomb[0][1]))
            if field[bomb[0][0] + 1, bomb[0][1]] != -1:
                for i in range(1, 4):
                    explosion_fields_in2.append((bomb[0][0] + i, bomb[0][1]))
            if field[bomb[0][0], bomb[0][1] - 1] != -1:
                for i in range(1, 4):
                    explosion_fields_in2.append((bomb[0][0], bomb[0][1] - i))
            if field[bomb[0][0], bomb[0][1] + 1] != -1:
                for i in range(1, 4):
                    explosion_fields_in2.append((bomb[0][0], bomb[0][1] + i))
        elif bomb[1] == 2:
            explosion_fields_in3.append(bomb[0])
            if field[bomb[0][0] - 1, bomb[0][1]] != -1:
                for i in range(1, 4):
                    explosion_fields_in3.append((bomb[0][0] - i, bomb[0][1]))
            if field[bomb[0][0] + 1, bomb[0][1]] != -1:
                for i in range(1, 4):
                    explosion_fields_in3.append((bomb[0][0] + i, bomb[0][1]))
            if field[bomb[0][0], bomb[0][1] - 1] != -1:
                for i in range(1, 4):
                    explosion_fields_in3.append((bomb[0][0], bomb[0][1] - i))
            if field[bomb[0][0], bomb[0][1] + 1] != -1:
                for i in range(1, 4):
                    explosion_fields_in3.append((bomb[0][0], bomb[0][1] + i))
        elif bomb[1] == 3:
            explosion_fields_in4.append(bomb[0])
            if field[bomb[0][0] - 1, bomb[0][1]] != -1:
                for i in range(1, 4):
                    explosion_fields_in4.append((bomb[0][0] - i, bomb[0][1]))
            if field[bomb[0][0] + 1, bomb[0][1]] != -1:
                for i in range(1, 4):
                    explosion_fields_in4.append((bomb[0][0] + i, bomb[0][1]))
            if field[bomb[0][0], bomb[0][1] - 1] != -1:
                for i in range(1, 4):
                    explosion_fields_in4.append((bomb[0][0], bomb[0][1] - i))
            if field[bomb[0][0], bomb[0][1] + 1] != -1:
                for i in range(1, 4):
                    explosion_fields_in4.append((bomb[0][0], bomb[0][1] + i))

    reachable_in1 = []
    reachable_in2 = []
    reachable_in3 = []
    reachable_in4 = []

    reachable_in1_moves = []
    reachable_in2_moves = []
    reachable_in3_moves = []
    reachable_in4_moves = []

    possible_Actions = possible_actions(game_state)
    if "UP" in possible_Actions:
        reachable_in1.append((position[0], position[1] - 1))
        reachable_in1_moves.append("UP")
        move_sets, fields = reachable_spaces_moves(
            game_state, (position[0], position[1] - 1), steps=1)
        reachable_in2.extend(fields)
        for moves in move_sets:
            moves.append("UP")
            reachable_in2_moves.append(moves)
        move_sets, fields = reachable_spaces_moves(
            game_state, (position[0], position[1] - 1), steps=2)
        reachable_in3.extend(fields)
        for moves in move_sets:
            moves.append("UP")
            reachable_in3_moves.append(moves)
        move_sets, fields = reachable_spaces_moves(
            game_state, (position[0], position[1] - 1), steps=3)
        reachable_in4.extend(fields)
        for moves in move_sets:
            moves.append("UP")
            reachable_in4_moves.append(moves)
    if "RIGHT" in possible_Actions:
        reachable_in1.append((position[0] + 1, position[1]))
        reachable_in1_moves.append("RIGHT")
        move_sets, fields = reachable_spaces_moves(
            game_state, (position[0] + 1, position[1]), steps=1)
        reachable_in2.extend(fields)
        for moves in move_sets:
            moves.append("RIGHT")
            reachable_in2_moves.append(moves)
        move_sets, fields = reachable_spaces_moves(
            game_state, (position[0] + 1, position[1]), steps=2)
        reachable_in3.extend(fields)
        for moves in move_sets:
            moves.append("RIGHT")
            reachable_in3_moves.append(moves)
        move_sets, fields = reachable_spaces_moves(
            game_state, (position[0] + 1, position[1]), steps=3)
        reachable_in4.extend(fields)
        for moves in move_sets:
            moves.append("RIGHT")
            reachable_in4_moves.append(moves)
    if "DOWN" in possible_Actions:
        reachable_in1.append((position[0], position[1] + 1))
        reachable_in1_moves.append("DOWN")
        move_sets, fields = reachable_spaces_moves(
            game_state, (position[0], position[1] + 1), steps=1)
        reachable_in2.extend(fields)
        for moves in move_sets:
            moves.append("DOWN")
            reachable_in2_moves.append(moves)
        move_sets, fields = reachable_spaces_moves(
            game_state, (position[0], position[1] + 1), steps=2)
        reachable_in3.extend(fields)
        for moves in move_sets:
            moves.append("DOWN")
            reachable_in3_moves.append(moves)
        move_sets, fields = reachable_spaces_moves(
            game_state, (position[0], position[1] + 1), steps=3)
        reachable_in4.extend(fields)
        for moves in move_sets:
            moves.append("DOWN")
            reachable_in4_moves.append(moves)
    if "LEFT" in possible_Actions:
        reachable_in1.append((position[0] - 1, position[1]))
        reachable_in1_moves.append("LEFT")
        move_sets, fields = reachable_spaces_moves(
            game_state, (position[0] - 1, position[1]), steps=1)
        reachable_in2.extend(fields)
        for moves in move_sets:
            moves.append("LEFT")
            reachable_in2_moves.append(moves)
        move_sets, fields = reachable_spaces_moves(
            game_state, (position[0] - 1, position[1]), steps=2)
        reachable_in3.extend(fields)
        for moves in move_sets:
            moves.append("LEFT")
            reachable_in3_moves.append(moves)
        move_sets, fields = reachable_spaces_moves(
            game_state, (position[0] - 1, position[1]), steps=3)
        reachable_in4.extend(fields)
        for moves in move_sets:
            moves.append("LEFT")
            reachable_in4_moves.append(moves)
    if "WAIT" in possible_Actions:
        reachable_in1.append((position[0], position[1]))
        reachable_in1_moves.append("WAIT")
        move_sets, fields = reachable_spaces_moves(
            game_state, (position[0], position[1]), steps=1)
        reachable_in2.extend(fields)
        for moves in move_sets:
            moves.append("WAIT")
            reachable_in2_moves.append(moves)
        move_sets, fields = reachable_spaces_moves(
            game_state, (position[0], position[1]), steps=2)
        reachable_in3.extend(fields)
        for moves in move_sets:
            moves.append("WAIT")
            reachable_in3_moves.append(moves)
        move_sets, fields = reachable_spaces_moves(
            game_state, (position[0], position[1]), steps=3)
        reachable_in4.extend(fields)
        for moves in move_sets:
            moves.append("WAIT")
            reachable_in4_moves.append(moves)

    save_space_in1 = []
    save_space_in2 = []
    save_space_in3 = []
    save_space_in4 = []

    save_moves_in1 = []
    save_moves_in2 = []
    save_moves_in3 = []
    save_moves_in4 = []

    for i in range(len(reachable_in1)):
        if reachable_in1[i] not in explosion_fields_in1:
            save_space_in1.append(reachable_in1[i])
            save_moves_in1.append(reachable_in1_moves[i])
    for i in range(len(reachable_in2)):
        if reachable_in2[i] not in explosion_fields_in1 and reachable_in2[i] not in explosion_fields_in2:
            save_space_in2.append(reachable_in2[i])
            save_moves_in2.append(reachable_in2_moves[i])
    for i in range(len(reachable_in3)):
        if reachable_in3[i] not in explosion_fields_in2 and reachable_in3[i] not in explosion_fields_in3:
            save_space_in3.append(reachable_in3[i])
            save_moves_in3.append(reachable_in3_moves[i])
    for i in range(len(reachable_in4)):
        if reachable_in4[i] not in explosion_fields_in3 and reachable_in4[i] not in explosion_fields_in4:
            save_space_in4.append(reachable_in4[i])
            save_moves_in4.append(reachable_in4_moves[i])

    save_actions_in1 = []
    save_actions_in2 = []
    save_actions_in3 = []
    save_actions_in4 = []

    save_actions_in1 = save_moves_in1
    for move2 in save_moves_in2:
        if move2[-1] in save_actions_in1:
            save_actions_in2.append(move2)

    for move3 in save_moves_in3:
        for move2 in save_actions_in2:
            if move3[-1] == move2[-1] and move3[-2] == move2[-2]:
                save_actions_in3.append(move3)

    for move4 in save_moves_in4:
        for move3 in save_actions_in3:
            if move4[-1] == move3[-1] and move4[-2] == move3[-2] and move4[-3] == move3[-3]:
                save_actions_in4.append(move4)

    others_pos = []
    for other in others:
        others_pos.append(other[3])

    up_save = 0
    right_save = 0
    down_save = 0
    left_save = 0
    wait_save = 0

    weight = 1

    if len(save_actions_in4) != 0:
        for moves in save_actions_in4:
            weight = 1
            current_pos = position
            for move in moves:
                if move == "UP":
                    current_pos = (current_pos[0], current_pos[1] - 1)
                if move == "RIGHT":
                    current_pos = (current_pos[0] + 1, current_pos[1])
                if move == "DOWN":
                    current_pos = (current_pos[0], current_pos[1] + 1)
                if move == "LEFT":
                    current_pos = (current_pos[0] - 1, current_pos[1])
                if move == "WAIT":
                    current_pos = (current_pos[0], current_pos[1])
                if (current_pos[0], current_pos[1] - 1) in others_pos or (current_pos[0], current_pos[1] + 1) in others_pos or (current_pos[0], current_pos[1] + 1) in others_pos or (current_pos[0] - 1, current_pos[1]) in others_pos or (current_pos[0], current_pos[1]) in others_pos:
                    weight = 0.5

            if moves[-1] == "UP":
                up_save = weight
            if moves[-1] == "RIGHT":
                right_save = weight
            if moves[-1] == "DOWN":
                down_save = weight
            if moves[-1] == "LEFT":
                left_save = weight
            if moves[-1] == "WAIT":
                wait_save = weight
    elif len(save_actions_in3) != 0:
        for moves in save_actions_in3:
            weight = 1
            current_pos = position
            for move in moves:
                if move == "UP":
                    current_pos = (current_pos[0], current_pos[1] - 1)
                if move == "RIGHT":
                    current_pos = (current_pos[0] + 1, current_pos[1])
                if move == "DOWN":
                    current_pos = (current_pos[0], current_pos[1] + 1)
                if move == "LEFT":
                    current_pos = (current_pos[0] - 1, current_pos[1])
                if move == "WAIT":
                    current_pos = (current_pos[0], current_pos[1])
                if (current_pos[0], current_pos[1] - 1) in others_pos or (current_pos[0], current_pos[1] + 1) in others_pos or (current_pos[0], current_pos[1] + 1) in others_pos or (current_pos[0] - 1, current_pos[1]) in others_pos or (current_pos[0], current_pos[1]) in others_pos:
                    weight = 0.5

            if moves[-1] == "UP":
                up_save = weight
            if moves[-1] == "RIGHT":
                right_save = weight
            if moves[-1] == "DOWN":
                down_save = weight
            if moves[-1] == "LEFT":
                left_save = weight
            if moves[-1] == "WAIT":
                wait_save = weight
    elif len(save_actions_in2) != 0:
        for moves in save_actions_in2:
            weight = 1
            current_pos = position
            for move in moves:
                if move == "UP":
                    current_pos = (current_pos[0], current_pos[1] - 1)
                if move == "RIGHT":
                    current_pos = (current_pos[0] + 1, current_pos[1])
                if move == "DOWN":
                    current_pos = (current_pos[0], current_pos[1] + 1)
                if move == "LEFT":
                    current_pos = (current_pos[0] - 1, current_pos[1])
                if move == "WAIT":
                    current_pos = (current_pos[0], current_pos[1])
                if (current_pos[0], current_pos[1] - 1) in others_pos or (current_pos[0], current_pos[1] + 1) in others_pos or (current_pos[0], current_pos[1] + 1) in others_pos or (current_pos[0] - 1, current_pos[1]) in others_pos or (current_pos[0], current_pos[1]) in others_pos:
                    weight = 0.5

            if moves[-1] == "UP":
                up_save = weight
            if moves[-1] == "RIGHT":
                right_save = weight
            if moves[-1] == "DOWN":
                down_save = weight
            if moves[-1] == "LEFT":
                left_save = weight
            if moves[-1] == "WAIT":
                wait_save = weight
    elif len(save_actions_in1) != 0:
        for moves in save_actions_in1:
            weight = 1
            current_pos = position
            for move in moves:
                if move == "UP":
                    current_pos = (current_pos[0], current_pos[1] - 1)
                if move == "RIGHT":
                    current_pos = (current_pos[0] + 1, current_pos[1])
                if move == "DOWN":
                    current_pos = (current_pos[0], current_pos[1] + 1)
                if move == "LEFT":
                    current_pos = (current_pos[0] - 1, current_pos[1])
                if move == "WAIT":
                    current_pos = (current_pos[0], current_pos[1])
                if (current_pos[0], current_pos[1] - 1) in others_pos or (current_pos[0], current_pos[1] + 1) in others_pos or (current_pos[0], current_pos[1] + 1) in others_pos or (current_pos[0] - 1, current_pos[1]) in others_pos or (current_pos[0], current_pos[1]) in others_pos:
                    weight = 0.5

            if moves[-1] == "UP":
                up_save = weight
            if moves[-1] == "RIGHT":
                right_save = weight
            if moves[-1] == "DOWN":
                down_save = weight
            if moves[-1] == "LEFT":
                left_save = weight
            if moves[-1] == "WAIT":
                wait_save = weight

    channels = []
    channels = np.append(channels, up_save)
    channels = np.append(channels, right_save)
    channels = np.append(channels, down_save)
    channels = np.append(channels, left_save)
    channels = np.append(channels, wait_save)
    stacked_channels = np.stack(channels)
    return stacked_channels.reshape(-1)


def state_to_features_bomb_old(game_state: dict) -> np.array:
    if game_state is None:
        return None

    position = game_state['self'][3]
    field = game_state['field']
    bombs = game_state['bombs']
    explosions = game_state['explosion_map']
    others = game_state['others']
    board_size = field.shape[0]

    others_pos = []
    for other in range(4):
        if len(others) > other:
            others_pos.append(others[other][3])
        else:
            others_pos.append((-1, -1))

    # 25 fields around agent
    small_field = []
    for i in range(-2, 3):
        for j in range(-2, 3):
            if position[0] + i <= 0:
                small_field.append(1)
            elif position[0] + i >= board_size:
                small_field.append(1)
            elif position[1] + j <= 0:
                small_field.append(1)
            elif position[1] + j >= board_size:
                small_field.append(1)
            else:
                if np.abs(field[position[0] + i, position[1] + j]) == 1:
                    small_field.append(1)
                elif explosions[position[0] + i, position[1] + j] > 0:
                    small_field.append(1)
                elif others_pos[0][0] == position[0] + i and others_pos[0][1] == position[1] + j:
                    small_field.append(1)
                elif others_pos[1][0] == position[0] + i and others_pos[1][1] == position[1] + j:
                    small_field.append(1)
                elif others_pos[2][0] == position[0] + i and others_pos[2][1] == position[1] + j:
                    small_field.append(1)
                elif others_pos[3][0] == position[0] + i and others_pos[3][1] == position[1] + j:
                    small_field.append(1)
                else:
                    small_field.append(0)

    # 8 entries, x and y dist to all bombs
    bomb_dists = []
    for bomb in range(4):
        if len(bombs) > bomb:
            bomb_dists.append(bombs[bomb][0][0] - position[0])
            bomb_dists.append(bombs[bomb][0][1] - position[1])
        else:
            bomb_dists.append(board_size)
            bomb_dists.append(board_size)

    # 4 timers of bombs
    bomb_times = []
    for bomb in range(4):
        if len(bombs) > bomb:
            bomb_times.append(bombs[bomb][1])
        else:
            bomb_times.append(20)

    # 8 entries, x and y dist to all others
    # others_dists = []
    # for other in range(4):
    #     if len(others) > other:
    #         others_dists.append(others[other][3][0] - position[0])
    #         others_dists.append(others[other][3][1] - position[1])
    #     else:
    #         others_dists.append(board_size)
    #         others_dists.append(board_size)

    channels = []
    channels = np.append(channels, small_field)
    channels = np.append(channels, bomb_dists)
    channels = np.append(channels, bomb_times)
    #channels = np.append(channels, others_dists)
    stacked_channels = np.stack(channels)
    return stacked_channels.reshape(-1)


def state_to_features_old(game_state: dict) -> np.array:
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

    coins = game_state['coins']
    position = game_state['self'][3]
    field = game_state['field']
    bombs = game_state['bombs']
    explosions = game_state['explosion_map']

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

    if len(bombs) == 0:
        min_xDist_bomb = 20
        min_yDist_bomb = 20
        bomb_timer = 400
    else:
        min_xDist_bomb = bombs[0][0][0] - position[0]
        min_yDist_bomb = bombs[0][0][1] - position[1]
        bomb_timer = bombs[0][1]

    isUpValid = field[position[1] - 1, position[0]]
    isRightValid = field[position[1], position[0] + 1]
    isDownValid = field[position[1] + 1, position[0]]
    isLeftValid = field[position[1], position[0] - 1]

    isUpExplosion = explosions[position[0], position[1] - 1]
    isRightExplosion = explosions[position[0] + 1, position[1]]
    isDownExplosion = explosions[position[0], position[1] + 1]
    isLeftExplosion = explosions[position[0] - 1, position[1]]

    channels = [min_xDist_coin, min_yDist_coin, isUpValid, isRightValid, isDownValid, isLeftValid, isDownExplosion,
                isUpExplosion, isRightExplosion, isLeftExplosion, min_xDist_bomb, min_yDist_bomb, bomb_timer]
    stacked_channels = np.stack(channels)
    return stacked_channels.reshape(-1)
