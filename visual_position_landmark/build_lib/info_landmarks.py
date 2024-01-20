import numpy as np
## chuyển đổi vị trí robot trong hệ tọa độ thứ i
def info_landmarks(tag_id):
    count = tag_id
    T = None  # Gán giá trị mặc định cho biến T
    r_O_i = None
    phi = None
    if count == 8:
        phi = 0
        c = np.cos(phi)
        s = np.sin(phi)
        T = [[c, -s, 0, 0],
            [s, c, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]]
        r_O_i = [[0], [0],[0],[1]]
    elif count == 9:
        phi = 0
        c = np.cos(phi)
        s = np.sin(phi)
        T = [[c, -s, 0, 0],
            [s, c, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]]
        r_O_i = [[200], [0],[0],[1]]
    elif count == 10:
        phi = 0
        c = np.cos(phi)
        s = np.sin(phi)
        T = [[c, -s, 0, 0],
            [s, c, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]]
        r_O_i = [[400], [0],[0],[1]]
    elif count == 11:
        phi = 0
        c = np.cos(phi)
        s = np.sin(phi)
        T = [[c, -s, 0, 0],
            [s, c, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]]
        r_O_i = [[600], [0],[0],[1]]
    elif count == 19:
        phi = 0
        c = np.cos(phi)
        s = np.sin(phi)
        T = [[c, -s, 0, 0],
            [s, c, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]]
        r_O_i = [[0], [0],[0],[1]] 
    elif count == 18:
        phi = 0
        c = np.cos(phi)
        s = np.sin(phi)
        T = [[c, -s, 0, 0],
            [s, c, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]]
        r_O_i = [[800], [0],[0],[1]] 
    elif count == 17:
        phi = 0
        c = np.cos(phi)
        s = np.sin(phi)
        T = [[c, -s, 0, 0],
            [s, c, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]]
        r_O_i = [[960], [0],[0],[1]] 
    elif count == 16:
        phi = 0
        c = np.cos(phi)
        s = np.sin(phi)
        T = [[c, -s, 0, 0],
            [s, c, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]]
        r_O_i = [[1120], [0],[0],[1]]
    elif count == 14:
        phi = 0
        c = np.cos(phi)
        s = np.sin(phi)
        T = [[c, -s, 0, 0],
            [s, c, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]]
        r_O_i = [[1280], [0],[0],[1]]

    return T, r_O_i, phi