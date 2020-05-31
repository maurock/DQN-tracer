from py_smallpt_newactions import main
from utils import create_log
from GPyOpt.methods import BayesianOptimization
import datetime

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
params['samples_training'] = 8
params['samples_test'] = 128

# Options
params['select_max_Q'] = True
params['select_average_Q'] = not params['select_max_Q']
params['scene'] = '3'
params['num_object'] = 18 if (params['scene'] == '1' or params['scene'] == '2') else 19
params['gaussian_kernel'] = True
params['training'] = False
params['test'] = True
params['limit_training'] = 16300 # int(0.8*params['w_training']*params['h_training']) #16300
params['kernel_size'] = 16
params['bayesOpt'] = False
params['double_action'] = False
params['Q_Learning'] = False
params['discretize_state'] = False
params['equally_sized_patches'] = False

# Folders
params['reference_path'] = 'images\\reference\\reference_scene3_256x256_5120spp.png'
params['path_SSIM_total'] = 'logs\\SSIM_total_' + str(datetime.datetime.now().strftime("%Y%m%d%H%M%S")) +'.txt'


class BayesianOptimizer():
    def __init__(self, params):
        self.params = params

    def optimize_raytracer(self):
        def optimize(inputs):
            print("INPUT", inputs)
            inputs = inputs[0]

            # Variables to optimize
            self.params["learning_rate"] = inputs[0]
            lr_string = '{:.8f}'.format(self.params["learning_rate"])[2:]
            self.params["dense_layer12"] = int(inputs[1])
            self.params["dense_layer3"] = int(inputs[2])
            self.params["state_layer1"] = int(inputs[3])
            self.params["state_layer2"] = int(inputs[4])
            self.params["advantage_layer1"] = int(inputs[5])
            self.params["advantage_layer2"] = int(inputs[6])
            self.params["epsilon_decay_linear"] = int(inputs[7])

            params['img_title'] = 'DDQN_scene{}_lr{}_struct{}-{}_{}-{}_{}-{}_eps{}'.format(params['scene'],
                                                                                        lr_string,
                                                                                        params['dense_layer12'],
                                                                                        params['dense_layer3'],
                                                                                        params['state_layer1'],
                                                                                        params['state_layer2'],
                                                                                        params['advantage_layer1'],
                                                                                        params['advantage_layer2'],
                                                                                        params['epsilon_decay_linear'])
            params['weight'] = 'weights_scene{}_'.format(params['scene']) + params['img_title'] + '.h5'
            params['training'] = True
            print(self.params)
            ssim = main(self.params)
            self.counter += 1
            return ssim

        self.counter = 0
        optim_params = [
            {"name": "learning_rate", "type": "continuous", "domain": (0.000040, 0.000005)},
            {"name": "dense_layer12", "type": "discrete", "domain": (100,200,300,400,500,600,700,800,900,1000)},
            {"name": "dense_layer3", "type": "discrete","domain": (100, 200, 300, 400, 500, 600, 700, 800, 900, 1000)},
            {"name": "state_layer1", "type": "discrete", "domain": (100, 200, 300, 400)},
            {"name": "state_layer2", "type": "discrete", "domain": (100, 200, 300, 400)},
            {"name": "advantage_layer1", "type": "discrete", "domain": (300, 400, 500,600,700)},
            {"name": "advantage_layer2", "type": "discrete", "domain": (300, 400, 500, 600, 700)},
            {"name":'epsilon_decay_linear', "type": "discrete", "domain": (3000,4000,5000,6000,7000)}
        ]

        bayes_optimizer = BayesianOptimization(f=optimize,
                                               domain=optim_params,
                                               initial_design_numdata=8,
                                               acquisition_type="EI",
                                               exact_feval=True,
                                               maximize=True)

        bayes_optimizer.run_optimization(max_iter=20)
        print('Optimized learning rate: ', bayes_optimizer.x_opt[0])
        print('Optimized dense layer12: ', bayes_optimizer.x_opt[1])
        print('Optimized dense layer3: ', bayes_optimizer.x_opt[2])
        print('Optimized state layer1: ', bayes_optimizer.x_opt[3])
        print('Optimized state layer2: ', bayes_optimizer.x_opt[4])
        print('Optimized advantage layer1: ', bayes_optimizer.x_opt[5])
        print('Optimized advantage layer2: ', bayes_optimizer.x_opt[6])
        print('Optimized epsilon linear decay: ', bayes_optimizer.x_opt[7])

        with open(params['path_SSIM_total'], 'a') as file:
            file.write("Best parameters: \n")
            file.write('Optimized learning rate: ' + bayes_optimizer.x_opt[0] + "\n")
            file.write('Optimized dense layer12: ' + bayes_optimizer.x_opt[1] + "\n")
            file.write('Optimized dense layer3: ' + bayes_optimizer.x_opt[2] + "\n")
            file.write('Optimized state layer1: ' + bayes_optimizer.x_opt[3] + "\n")
            file.write('Optimized state layer2: ' + bayes_optimizer.x_opt[4] + "\n")
            file.write('Optimized advantage layer1: ' + bayes_optimizer.x_opt[5] + "\n")
            file.write('Optimized advantage layer2: ' + bayes_optimizer.x_opt[6] + "\n")
            file.write('Optimized epsilon linear decay: ' + bayes_optimizer.x_opt[7])
        return self.params

##################
#      Main      #
##################
if __name__ == '__main__':

    # Traditional training and testing
    if params['bayesOpt'] == False:
        # Set automatic parameters
        lr_string = '{:.8f}'.format(params["learning_rate"])[2:]
        params['img_title'] = 'DDQN_scene{}_lr{}_struct{}-{}_{}-{}_{}-{}_eps{}_DQN'.format(params['scene'],
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




