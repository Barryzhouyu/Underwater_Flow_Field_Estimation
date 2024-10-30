def calculate_navigation_time(pos, cell_length_x, cell_length_y, v_x, v_y, F_x, F_y):
    s_tr_x = v_x + F_x
    s_tr_y = v_y + F_y
    t_xy = []

    if s_tr_x < 0:
        t_x = (cell_length_x - (pos[0] % cell_length_x)) / -s_tr_x
    else:
        t_x = (cell_length_x - (pos[0] % cell_length_x)) / s_tr_x

    if s_tr_y < 0:
        t_y = (pos[1] % cell_length_y) / -s_tr_y
    else:
        t_y = (cell_length_y - (pos[1] % cell_length_y)) / s_tr_y

    t_xy.append(t_x)
    t_xy.append(t_y)
    t_min = min(t_x, t_y)
    return t_min, t_xy