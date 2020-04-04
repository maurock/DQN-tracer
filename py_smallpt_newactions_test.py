from utils import write_ppm
import random
from smallpt_pybind import *
import numpy as np
from DQNAgent_DDQN import *
import copy
import time

# Scene
REFRACTIVE_INDEX_OUT = 1.0
REFRACTIVE_INDEX_IN = 1.5
NUMBER_OBJ = 18
limit = 10000  # parameter for training
kernel_size = 16
state_space = 51
action_space = 72
arr_line = np.linspace(0, 1, kernel_size)
# py-smallpt3-10000L2000-72actions-lineareps-lr00001-1000nnx4-KERNEL16-targetnet-32spp-256-depth6-noisereduction-AGAIN

title = "relu-nnx4-newAction-maxAction-eps5000-DDQN_struct4-"
img_title = "py-smallpt-lr00001-" + title
weight_title = "weights-lr00001-" + title + ".h5"

# weights_path = weight_title                                                                         # when training + active in the same session
weights_path = 'weights-lr00001-relu-nnx4-newAction-maxAction-eps5000-DDQN_struct4-.h5'  # when only active

learning = False
cos_weight_is = False


class Counter:
    def __init__(self):
        self.full_count = 1
        self.reward_count = 0


def gaussian(x, mu, sig):
    return 1. / (math.sqrt(2. * math.pi) * sig) * np.exp(-np.power((x - mu) / sig, 2.) / 2)


@jit(nopython=True)
def create_state(x, y, z, nlx, nly, nlz):
    def gaussian(x, mu, sig):
        return 1. / (math.sqrt(2. * math.pi) * sig) * np.exp(-np.power((x - mu) / sig, 2.) / 2)

    gaussiana_x = gaussian(arr_line, x / 99, 1 / kernel_size)
    gaussiana_x[np.abs(gaussiana_x) < 0.01] = 0
    gaussiana_y = gaussian(arr_line, y / 82, 1 / kernel_size)
    gaussiana_y[np.abs(gaussiana_y) < 0.01] = 0
    gaussiana_z = gaussian(arr_line, z / 170, 1 / kernel_size)
    gaussiana_z[np.abs(gaussiana_z) < 0.01] = 0
    return np.concatenate(
        (gaussiana_x, gaussiana_y, gaussiana_z, np.array([nlx, nly, nlz])), axis=0)


def store_temporary_memory_2(self, arr):
    self.temporary_memory.append(arr)


def remember_2(self):
    for i in range(len(self.temporary_memory)):
        if ((len(self.memory) + 1) > self.memory_length):
            del self.memory[0]
        self.memory.append(self.temporary_memory[i])



def visualize_Q_intensity(agent, state):
    arr = agent.model.predict(state.reshape(1, state_space))[0]
    summa = np.sum(arr)

    r = 1 - (0.49 * summa / 40)
    if (r < 0.51):
        r = 0.51
    g = 1 - summa / 40
    b = 1
    return Vec(r, g, b)


def plot_Q(x, nl, state, agent):
    if x.get_x() > 70 and x.get_x() < 75 and x.get_y() > 25 and x.get_y() < 30 and x.get_z() > 75 and x.get_z() < 80 and nl.get_x() == 0 and nl.get_y() == 1 and nl.get_z() == 0:
        p = Vec(72, 25.01, 78)
        nl = Vec(0, 1, 0)
        arr_line = np.linspace(0, 1, kernel_size)
        print(arr_line)
        gaussiana_x = gaussian(arr_line, p.get_x() / 99, 1 / kernel_size)
        gaussiana_x[np.abs(gaussiana_x) < 0.01] = 0
        gaussiana_y = gaussian(arr_line, p.get_y() / 82, 1 / kernel_size)
        gaussiana_y[np.abs(gaussiana_y) < 0.01] = 0
        gaussiana_z = gaussian(arr_line, p.get_z() / 170, 1 / kernel_size)
        gaussiana_z[np.abs(gaussiana_z) < 0.01] = 0
        target_state = np.concatenate(
            (gaussiana_x, gaussiana_y, gaussiana_z, np.array([nl.get_x(), nl.get_y(), nl.get_z()])), axis=0)
        prediction = agent.model.predict(target_state.reshape(1, state_space))[0]

        # prediction = agent.model.predict(state.reshape(1, state_space))[0]
        file1 = open("plot_Q_dqn_128_2.txt", "a")
        for i in range(0, 71):
            file1.write(str(prediction[i]) + ",")
        file1.write(str(prediction[71]) + "\n")


def radiance(ray, depth, dict, agent, count, counter_bouncer):
    L = Vec()
    F = Vec(1.0, 1.0, 1.0)
    hitobj = Hit_record()
    hitobj.set_BRDF(1)
    hitobj.set_prob(1)
    hitobj.set_costheta(1)
    temp_mem = []

    # To pyplot for DEBUG
    p_copy = Vec(0, 0, 0)
    nl = Vec(0, 0, 0)

    while True:
        done = 0
        p = hittingPoint(ray, hitobj, NUMBER_OBJ)
        if (not hitobj.get_hit()):
            print("HAPPENED")
            return Vec()

        counter_bouncer.full_count += 1

        if depth > 5 and hitobj.get_e().get_x() < 1:
            continue_probability = F.get_max()
            if random.random() >= continue_probability:
                return L
            F = F * (1 / continue_probability)
        depth += 1

        '''
        # Print scattered rays with pyplot DEBUG
        if p_copy.get_x() > 70 and p_copy.get_x() < 75 and p_copy.get_y() > 20 and p_copy.get_y() < 30 and p_copy.get_z() > 75 and p_copy.get_z() < 80 and nl.get_x() == 0 and nl.get_y() == 1 and nl.get_z() == 0:
            with open('plotly/scattered_rays_target_dqn.txt', 'a') as scatterrays:
                stringa = str(p.get_x()) + "," + str(p.get_y()) + "," + str(p.get_z()) + "\n"
                scatterrays.write(stringa)
            with open('plotly/scattered_state_dqn_MaxAction_temp.txt', 'a') as scatterstate:
                stringa = str(p_copy.get_x()) + "," + str(p_copy.get_y()) + "," + str(p_copy.get_z()) + "\n"
                scatterstate.write(stringa)
        p_copy = p
        '''

        reward = hitobj.get_e().get_x()
        hitobj.set_oldid(hitobj.get_id())
        hitobj.set_hit(False)
        if cos_weight_is == True:
            hitobj.set_prob(1)
            hitobj.set_BRDF(1)
            hitobj.set_costheta(1)

        L = L + (F.mult(hitobj.get_e()))

        nl = hitobj.get_nl()

        hitobj.set_BRDF((hitobj.get_c().get_max()) / math.pi)

        # hitobj.set_costheta(math.fabs(nl.dot(ray.d)))
        if hitobj.get_e().get_x() > 5:
            counter_bouncer.reward_count += 1
            done = 1
            return L
        arr_line = np.linspace(0, 1, kernel_size)
        gaussiana_x = gaussian(arr_line, p.get_x() / 99, 1 / kernel_size)
        gaussiana_x[np.abs(gaussiana_x) < 0.01] = 0
        gaussiana_y = gaussian(arr_line, p.get_y() / 82, 1 / kernel_size)
        gaussiana_y[np.abs(gaussiana_y) < 0.01] = 0
        gaussiana_z = gaussian(arr_line, p.get_z() / 170, 1 / kernel_size)
        gaussiana_z[np.abs(gaussiana_z) < 0.01] = 0
        next_state = np.concatenate(
            (gaussiana_x, gaussiana_y, gaussiana_z, np.array([nl.get_x(), nl.get_y(), nl.get_z()])), axis=0)

        if done:
            return L

        action_int = agent.do_action(next_state, hitobj, dict)
        F = F.mult(hitobj.get_c()) * hitobj.get_prob() * hitobj.get_BRDF() * hitobj.get_costheta()
        ray = Ray(p, DQNScattering(dict, nl, action_int).norm())
        # ray = Ray(p,importanceSampling_scattering(nl))
        # state = copy.deepcopy(next_state)


import sys

if __name__ == "__main__":
    w = 256
    h = 256
    # samps_training = 3
    samps_scattering = 32
    cam = Camera(Vec(50, 40, 168), Vec(50, 40, 5), Vec(0, 1, 0), 65, w / h)

    L = Vec()
    i = 0

    Ls = np.empty([w * h, 3])
    Ls.fill(0)
    agent = DQN(limit)
    counter_bounces = Counter()
    zero_contrib = 0
    '''
    agent.exploration_rate = 0.04
    agent.model = agent.network("weights-10000L3500-72actions-lineareps-lr00001-1000nnx4-KERNEL16-targetnet.h5")
    '''

    if action_space == 24:
        log_sphere_point = "sphere_point.csv"
    elif action_space == 72:
        log_sphere_point = "sphere_point4.csv"
    dict = initialize_dictAction(log_sphere_point)

    if learning == False and cos_weight_is == False:
        agent.exploration_rate = 0
        agent.model = agent.network(weights_path)
    print("ACTIVE PHASEE")

    for s in range(0, samps_scattering):
        print('\rSample number: ', s)
        i = 0
        for y in range(h):
            # pixel row
            print('\rRendering ({0} spp) {1:0.2f}%'.format(samps_scattering, 100.0 * y / (
                        h - 1)))  # , 'reward ratio:', counter_bounces.reward_count/counter_bounces.full_count)
            for x in range(w):
                u = (x - 0.5 + random.random()) / w
                v = ((h - y - 1) - 0.5 + random.random()) / h
                d = cam.get_ray(u, v)
                rad = radiance(Ray(d.o, d.d.norm()), 0, dict, agent, i, counter_bounces)
                L += rad * (1.0 / samps_scattering)
                Ls[i][0] = Ls[i][0] + clamp(L.get_x())
                Ls[i][1] = Ls[i][1] + clamp(L.get_y())
                Ls[i][2] = Ls[i][2] + clamp(L.get_z())
                L = Vec()
                i += 1


        img_title_temp = img_title + str(s) + ".ppm"
        write_ppm(w, h, Ls * samps_scattering / (s + 1), fname='img/' + img_title_temp)

        # write_ppm(w, h, Ls, fname=img_title)
    print("NUMBER OF BOUNCES: ", counter_bounces.full_count / (w * h * samps_scattering))
    write_ppm(w, h, Ls, fname=img_title)
