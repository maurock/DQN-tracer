import math
import numpy as np
from smallpt_pybind import *
import json
import copy

M_PI = 3.14159265358979323846

def clamp(x, low = 0.0, high = 1.0):
    if x > high:
        return high
    if x < low:
        return low
    return x

def to_byte(x, gamma = 2.2):
    return int(clamp(255.0 * pow(x, 1.0 / gamma), 0.0, 255.0))

def write_ppm(w, h, Ls, fname = "py-image.ppm"):
    with open(fname, 'w') as outfile:
        outfile.write('P3\n{0} {1}\n{2}\n'.format(w, h, 255));
        for L in Ls:
            outfile.write('{0} {1} {2} '.format(to_byte(L[0]), to_byte(L[1]), to_byte(L[2])));

def visualize_Q_intensity(agent, params):
    arr = agent.model.predict(state.reshape(1, params['state_space']))[0]
    summa = np.sum(arr)

    r = 1 - (0.49 * summa / 40)
    if (r < 0.51):
        r = 0.51
    g = 1 - summa / 40
    b = 1
    return Vec(r, g, b)


def plot_Q_allstates(x, reward, agent, state):
    file1 = open("plot_Q_dqn_debug.txt", "a")
    print("x.get_x()", str(x.get_x()))
    file1.write(str(reward) + ",")
    prediction = agent.model.predict(state.reshape(1, state_space))[0]
    for i in range(0, 71):
        file1.write(str(prediction[i]) + ",")
    file1.write(str(prediction[71]) + "\n")


def plot_Q(x, nl, state, agent, params):
    # if x.get_x() > 70 and x.get_x() < 75 and x.get_y() > 25 and x.get_y() < 30 and x.get_z()>75 and x.get_z()<80 and nl.get_x() == 0 and nl.get_y() == 1 and nl.get_z() == 0:
    p = Vec(72, 25.01, 78)
    nl = Vec(0, 1, 0)
    arr_line = np.linspace(0, 1, params['kernel_size'])
    gaussiana_x = gaussian(arr_line, p.get_x() / 99, 1 / params['kernel_size'])
    gaussiana_x[np.abs(gaussiana_x) < 0.01] = 0
    gaussiana_y = gaussian(arr_line, p.get_y() / 82, 1 / params['kernel_size'])
    gaussiana_y[np.abs(gaussiana_y) < 0.01] = 0
    gaussiana_z = gaussian(arr_line, p.get_z() / 170, 1 / params['kernel_size'])
    gaussiana_z[np.abs(gaussiana_z) < 0.01] = 0
    target_state = np.concatenate(
        (gaussiana_x, gaussiana_y, gaussiana_z, np.array([nl.get_x(), nl.get_y(), nl.get_z()])), axis=0)
    prediction = agent.model.predict(target_state.reshape(1, params['state_space']))[0]

    # prediction = agent.model.predict(state.reshape(1, state_space))[0]
    file1 = open("plot_Q_dqn_128-DQNvsADQN.txt", "a")
    for i in range(0, 71):
        file1.write(str(prediction[i]) + ",")
    file1.write(str(prediction[71]) + "\n")

def scattered_rays_plotly(p_copy):
    if p_copy.get_x() > 70 and p_copy.get_x() < 75 and p_copy.get_y() > 20 and p_copy.get_y() < 30 and p_copy.get_z() > 75 and p_copy.get_z() < 80 and nl.get_x() == 0 and nl.get_y() == 1 and nl.get_z() == 0:
        with open('plotly/scattered_rays_target_dqn.txt', 'a') as scatterrays:
            stringa = str(p.get_x()) + "," + str(p.get_y()) + "," + str(p.get_z()) + "\n"
            scatterrays.write(stringa)
        with open('plotly/scattered_state_dqn_MaxAction_temp.txt', 'a') as scatterstate:
            stringa = str(p_copy.get_x()) + "," + str(p_copy.get_y()) + "," + str(p_copy.get_z()) + "\n"
            scatterstate.write(stringa)
    p_copy = copy.deepcopy(p)

def gaussian(x, mu, sig):
    return 1. / (math.sqrt(2. * math.pi) * sig) * np.exp(-np.power((x - mu) / sig, 2.) / 2)

def create_log(params):
    with open('logs\\' + params['img_title'] + '.txt', 'w') as file:
        file.write(json.dumps(params['img_title'])+'\n\n')
        for _ in (list(params.keys())):
            file.write(str(_) + " : " + json.dumps(params[_]) + "\n")






