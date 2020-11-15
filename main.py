from py_smallpt import main
from utils import create_log
import datetime
from bayesian_optimizer import BayesianOptimizer
##################
# Set parameters #
##################
params = dict()

# Neural network
params['state_space'] = 51
params['action_space'] = 72
params['epsilon_decay_linear'] = 7500
params['learning_rate'] = 0.000008
params['dense_layer12'] = 500
params['dense_layer3'] = 700
params['state_layer1'] = 400
params['state_layer2'] = 400
params['advantage_layer1'] = 600
params['advantage_layer2'] = 700

# Rendering
params['w_training'] = 128
params['h_training'] = 128
params['w_test'] = 256
params['h_test'] = 256
params['samples_training'] = 1
params['samples_test'] = 12

# Options
params['select_max_Q'] = True
params['select_average_Q'] = not params['select_max_Q']
params['scene'] = '3'
params['num_object'] = 18 if (params['scene'] == '1' or params['scene'] == '2') else 19
params['gaussian_kernel'] = True
params['training'] = True
params['test'] = True
params['limit_training'] = 16300
params['kernel_size'] = 16
params['bayesOpt'] = False
params['double_action'] = False
params['Q_Learning'] = False
params['discretize_state'] = False
params['equally_sized_patches'] = False

# Folders
params['reference_path'] = 'images\\reference\\reference_scene3_256x256_5120spp.png'
params['path_SSIM_total'] = 'logs\\SSIM_total_' + str(datetime.datetime.now().strftime("%Y%m%d%H%M%S")) +'.txt'

##################
#      Main      #
##################
if __name__ == '__main__':

    # Traditional training and testing
    if params['bayesOpt'] == False:
        # Set automatic parameters
        lr_string = '{:.8f}'.format(params["learning_rate"])[2:]
        params['img_title'] = 'DDQN_scene{}_lr{}_struct{}-{}_{}-{}_{}-{}_eps{}_DQN_128' \
                              ''.format(params['scene'],
                                lr_string,
                                params['dense_layer12'],
                                params['dense_layer3'],
                                params['state_layer1'],
                                params['state_layer2'],
                                params['advantage_layer1'],
                                params['advantage_layer2'],
                                params['epsilon_decay_linear'])
        params['weight'] = 'weights_scene{}_'.format(params['scene']) + params['img_title'] + '.h5'

        # Custom weight
        params['weight'] = 'weights_scene3_DDQN_scene3_DQN_128.h5'
        main(params)

    # Bayesian Optimization
    else:
        bayesOpt = BayesianOptimizer(params)
        bayesOpt.optimize_raytracer()
