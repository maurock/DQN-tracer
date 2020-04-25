from utils import *
from smallpt_pybind import *
from DQNAgent_DDQN import *
import copy
from skimage.measure import compare_ssim
from PIL import Image
import numpy as np
from tqdm import tqdm
import pickle

class Counter:
    def __init__(self):
        self.full_count = 0
        self.reward_count = 0

def score(final_image, reference_image_path, params, s):
    ref = Image.open(reference_image_path).convert("RGB")
    ref_array = np.array(ref).reshape((params["w_test"], params["h_test"],3))
    for count, pixel in enumerate(final_image):
        final_image[count][0] = to_byte(pixel[0])
        final_image[count][1] = to_byte(pixel[1])
        final_image[count][2] = to_byte(pixel[2])
    final_image = np.array(final_image).astype(np.uint8).reshape((params["w_test"], params["h_test"],3))
    ssim = compare_ssim(ref_array, final_image, multichannel = True)
    with open('logs\\' + params['img_title'] + '.txt', 'a') as file:
        file.write("SSIM score (" + str(s) + " episode) : " + str(ssim) + "\n")
    return ssim

def radiance(ray, depth, dict_act, agent, count, counter_bouncer, params, dict_state):
    L = Vec()
    F = Vec(1.0, 1.0, 1.0)
    hitobj = Hit_record()
    hitobj.set_BRDF(1)
    hitobj.set_prob(1)
    hitobj.set_costheta(1)

    while True:
        done = 0
        p = hittingPoint(ray, hitobj, params['num_object'])

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

        reward = hitobj.get_e().get_x()
        hitobj.set_oldid(hitobj.get_id())
        hitobj.set_hit(False)

        L = L + (F.mult(hitobj.get_e()))

        nl = hitobj.get_nl()

        hitobj.set_BRDF((hitobj.get_c().get_max()) / math.pi)

        # hitobj.set_costheta(math.fabs(nl.dot(ray.d)))
        if hitobj.get_e().get_x() > 5:
            done = 1

        if params['gaussian_kernel'] == True:
            # Gaussian kernel
            arr_line = np.linspace(0, 1, params['kernel_size'])
            gaussiana_x = gaussian(arr_line, p.get_x() / 99, 1 / params['kernel_size'])
            gaussiana_x[np.abs(gaussiana_x) < 0.01] = 0
            gaussiana_y = gaussian(arr_line, p.get_y() / 82, 1 / params['kernel_size'])
            gaussiana_y[np.abs(gaussiana_y) < 0.01] = 0
            gaussiana_z = gaussian(arr_line, p.get_z() / 170, 1 / params['kernel_size'])
            gaussiana_z[np.abs(gaussiana_z) < 0.01] = 0
            next_state = np.concatenate((gaussiana_x, gaussiana_y, gaussiana_z, np.array([nl.get_x(), nl.get_y(), nl.get_z()])), axis=0)
        else:
            # Not Gaussian kernel
            next_state = np.array([p.get_x(), p.get_y(), p.get_z(), nl.get_x(), nl.get_y(), nl.get_z()])

        if params['training'] and not params['double_action']:
            if depth > 1 and count < params['limit_training']:
                q_value = agent.train(state, action_int, reward, next_state, done, hitobj.get_BRDF(), dict_act, nl, params)

                if params['discretize_state']:
                    # Discretize state
                    key = (math.ceil(p_old.get_x() / 5), math.ceil(p_old.get_y() / 5), math.ceil(p_old.get_z() / 5),
                                math.ceil(nl_old.get_x()), math.ceil(nl_old.get_y()), math.ceil(nl_old.get_z()))
                    scattering_dir_spher = cartToSpher(scattering_dir_cart)
                    if key in dict_state.keys():
                        dict_state[key] = np.concatenate([ dict_state[key], np.array([[int(action_int),scattering_dir_spher.get_y(),scattering_dir_spher.get_z(), q_value]])])
                    else:
                        dict_state[key] = np.array([[int(action_int),scattering_dir_spher.get_y(),scattering_dir_spher.get_z(),q_value]])


        # Only if double action
        elif params['double_action']:
            next_action_int, next_double_action_int, next_state_double_action = agent.do_action_double_action(next_state, hitobj, dict_act)
            if params['training']:
                if depth > 1 and count < params['limit_training']:
                    agent.train_double_action(state_double_action, action_int, double_action_int, reward,
                                              next_state_double_action, done, hitobj.get_BRDF(), dict_act, nl, params)
        if done:
            return L
        if not params['double_action']:
            action_int = agent.do_action(next_state, hitobj, dict_act)
            F = F.mult(hitobj.get_c()) * hitobj.get_prob() * hitobj.get_BRDF() * hitobj.get_costheta()
            scattering_dir_cart = DQNScattering(dict_act, nl, action_int, 0).norm()
            ray = Ray(p, scattering_dir_cart)
            state = copy.deepcopy(next_state)
            p_old = p
            nl_old = nl

        # Only if double action
        else:
            F = F.mult(hitobj.get_c()) * hitobj.get_prob() * hitobj.get_BRDF() * hitobj.get_costheta()
            ray = Ray(p, DQNScattering(dict_act, nl, next_action_int, next_double_action_int).norm())
            state = copy.deepcopy(next_state)
            state_double_action = copy.deepcopy(next_state_double_action)
            double_action_int = copy.deepcopy(next_double_action_int)
            action_int = copy.deepcopy(next_action_int)

def main(params):
    # Save params
    create_log(params)
    if params['action_space'] == 24:
        log_sphere_point = "sphere_point.csv"
    elif params['action_space'] == 72:
        log_sphere_point = "sphere_point4.csv"
    dict_act = initialize_dictAction(log_sphere_point)

    # Training
    if params['training']:
        print("Training phase.....")
        cam = Camera(Vec(50, 40, 168), Vec(50, 40, 5), Vec(0, 1, 0), 65, params['w_training'] / params['h_training'])
        L = Vec()
        Ls = np.empty([params['w_training'] * params['h_training'], 3])
        Ls.fill(0)
        agent = DQN(params)
        counter_bounces = Counter()
        dict_state = dict()
        i = 0
        for y in tqdm(range(params['h_training']), desc='Rendering ({0} spp), learning rate {1:11.8f}'.format(params['samples_training'],agent.learning_rate), position=0, leave=True):
            if i > params['limit_training']:
                break
            # pixel row
            #print('\rRendering ({0} spp) {1:0.2f}% learning rate {2:11.8f}'.format(params['samples_training'], 100.0 * y / (params['h_training'] - 1),agent.learning_rate))
            for x in range(params['w_training']):

                if i > params['limit_training']:
                    break
                for s in range(params['samples_training']):
                    u = (x - 0.5 + random.random()) / params['w_training']
                    v = ((params['h_training'] - y - 1) - 0.5 + random.random()) / params['h_training']
                    d = cam.get_ray(u, v)
                    rad = radiance(Ray(d.o, d.d.norm()), 0, dict_act, agent, i, counter_bounces, params, dict_state)
                    L += rad * (1.0 / params['samples_training'])
                Ls[i][0] = clamp(L.get_x())
                Ls[i][1] = clamp(L.get_y())
                Ls[i][2] = clamp(L.get_z())
                L = Vec()
                i += 1

                if i < params['limit_training']:
                    agent.model_target = agent.model
                    if agent.exploration_rate > agent.epsilon_min:
                        agent.exploration_rate = 1 - 1/agent.epsilon_decay_linear * i
                if i > params['limit_training']:
                    print("params[weight : ",params['weight'] )
                    agent.model.save_weights("weights\\" + params['weight'])
                    if params['double_action']:
                        agent.model_double_action.save_weights("weights\\" + params['weight_double_action'])
                    print("Weights saved...")
                    time.sleep(3)

                    with open('dict_state.p', 'wb') as fp:
                        pickle.dump(dict_state, fp, protocol=pickle.HIGHEST_PROTOCOL)

                    break
            #write_ppm(params['w_training'], params['h_training'], Ls, fname=img_title)
        #print("NUMBER OF BOUNCES: ", counter_bounces.full_count / (params['w_training'] * params['h_training'] * params['samples_training']))

    # Test
    if params['test']:
        print("Testing phase.....")
        params['training'] = False
        cam = Camera(Vec(50, 40, 168), Vec(50, 40, 5), Vec(0, 1, 0), 65, params['w_test'] / params['h_test'])
        L = Vec()
        Ls = np.empty([params['w_test'] * params['h_test'], 3])
        Ls.fill(0)
        agent = DQN(params)
        counter_bounces = Counter()
        agent.exploration_rate = 0
        weights_path = 'weights\\' + params['weight']
        agent.model = agent.network(weights_path)
        dict_state = dict()
        if params['double_action']:
            weights_path_double_action = 'weights\\' + params['weight_double_action']
            agent.model_double_action = agent.network_double_action(weights_path_double_action)
        agent.exploration_rate = 0
        print('\rRendering ({0} spp)'.format(params['samples_training']))
        for s in range(0, params['samples_test']):
            print('\rSample number: ', s)
            i = 0
            for y in tqdm(range(params['h_test']), position=0, leave=True):
                # pixel row
                #print('\rRendering ({0} spp) {1:0.2f}%'.format(params['samples_test'], 100.0 * y / (
                #        params['h_test'] - 1)))  # , 'reward ratio:', counter_bounces.reward_count/counter_bounces.full_count)
                for x in range(params['w_test']):
                    u = (x - 0.5 + random.random()) / params['w_test']
                    v = ((params['h_test'] - y - 1) - 0.5 + random.random()) / params['h_test']
                    d = cam.get_ray(u, v)
                    rad = radiance(Ray(d.o, d.d.norm()), 0, dict_act, agent, i, counter_bounces, params, dict_state)
                    L += rad * (1.0 / params['samples_test'])
                    Ls[i][0] = Ls[i][0] + clamp(L.get_x())
                    Ls[i][1] = Ls[i][1] + clamp(L.get_y())
                    Ls[i][2] = Ls[i][2] + clamp(L.get_z())
                    L = Vec()
                    i += 1
            final_image = Ls * params['samples_test'] / (s + 1)
            img_title_temp = 'images\\'+ params['img_title'] + '-' + str(s) + ".ppm"
            write_ppm(params['w_test'], params['h_test'], final_image, fname= img_title_temp)
            ssim = score(final_image, params['reference_path'], params=params, s=s)
            print("Score: ", ssim)

        print("NUMBER OF BOUNCES: ", counter_bounces.full_count / (params['w_test'] * params['h_test'] * params['samples_test']))
    with open(params['path_SSIM_total'], 'a') as file:
        file.write(str(params['img_title']) + ': ' + str(ssim) + '\n');
    return ssim

