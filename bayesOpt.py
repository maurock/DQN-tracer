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
params['epsilon_decay_linear'] = 5000
params['learning_rate'] = 0.00001
params['dense_layer'] = 1000
params['state_layer'] = 200
params['advantage_layer'] = 400

# Rendering
params['w_training'] = 24
params['h_training'] = 24
params['w_test'] = 128
params['h_test'] = 128
params['samples_training'] = 3
params['samples_test'] = 3

# Options
params['select_max_Q'] = True
params['select_average_Q'] = not params['select_max_Q']
params['scene'] = '3'
params['num_object'] = 18 if (params['scene'] == '1' or params['scene'] == '2') else 19
params['gaussian_kernel'] = True
params['training'] = True
params['test'] = True
params['limit_training'] = 100 #int(0.8*params['w_training']*params['h_training'])
params['kernel_size'] = 16
params['bayesOpt'] = True

# Folders
params['reference_path'] = 'images\\reference\\reference_scene3_128x128_5120spp.png'
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
            self.params["dense_layer"] = int(inputs[1])
            self.params["state_layer"] = int(inputs[2])
            self.params["advantage_layer"] = int(inputs[3])
            self.params["epsilon_decay_linear"] = int(inputs[4])

            params['img_title'] = 'DDQN_scene{}_lr{}_struct{}_{}_{}_eps{}'.format(params['scene'],
                                                                                      lr_string,
                                                                                      params['dense_layer'],
                                                                                      params['state_layer'],
                                                                                      params['advantage_layer'],
                                                                                      params['epsilon_decay_linear'])
            params['weight'] = 'weights_scene{}_'.format(params['scene']) + params['img_title'] + '.h5'
            params['training'] = True
            print(self.params)
            ssim = main(self.params)
            self.counter += 1
            return ssim

        self.counter = 0
        optim_params = [
            {"name": "learning_rate", "type": "continuous", "domain": (0.000001, 0.00005)},
            {"name": "dense_layer", "type": "discrete", "domain": (100,200,300,400,500,600,700,800,900,1000)},
            {"name": "state_layer", "type": "discrete", "domain": (100, 200, 300, 400, 500,600,700)},
            {"name": "advantage_layer", "type": "discrete", "domain": (100, 200, 300, 400, 500,600,700)},
            {"name":'epsilon_decay_linear', "type": "discrete", "domain": (2000,3000,4000,5000,6000,7000,8000,9000,10000)}
        ]

        bayes_optimizer = BayesianOptimization(f=optimize,
                                               domain=optim_params,
                                               initial_design_numdata=6,
                                               acquisition_type="EI",
                                               exact_feval=True,
                                               maximize=True)

        bayes_optimizer.run_optimization(max_iter=19)
        print('Optimized learning rate: ', bayes_optimizer.x_opt[0])
        print('Optimized dense layer: ', bayes_optimizer.x_opt[1])
        print('Optimized state layer: ', bayes_optimizer.x_opt[2])
        print('Optimized advantage layer: ', bayes_optimizer.x_opt[3])
        print('Optimized epsilon linear decay: ', bayes_optimizer.x_opt[4])

        with open(params['path_SSIM_total'], 'a') as file:
            file.write("Best parameters: \n")
            file.write('Optimized learning rate: ' + bayes_optimizer.x_opt[0] + "\n")
            file.write('Optimized dense layer: ' + bayes_optimizer.x_opt[1] + "\n")
            file.write('Optimized state layer: ' + bayes_optimizer.x_opt[2] + "\n")
            file.write('Optimized advantage layer: ' + bayes_optimizer.x_opt[3] + "\n")
            file.write('Optimized epsilon linear decay: ' + bayes_optimizer.x_opt[4])
        return self.params





##################
#      Main      #
##################
if __name__ == '__main__':

    # Traditional training and testing
    if params['bayesOpt'] == False:
        # Set automatic parameters
        lr_string = '{:.8f}'.format(params["learning_rate"])[2:]
        params['img_title'] = 'DDQN_scene{}_lr{}_struct{}_{}_{}_eps{}_NotOpt'.format(params['scene'],lr_string,
                                            params['dense_layer'],params['state_layer'],params['advantage_layer'],params['epsilon_decay_linear'])
        params['weight'] = 'weights_scene{}_'.format(params['scene']) + params['img_title'] + '.h5'

        # Custom weight
        #params['weight'] = 'weights_scene3_DDQN_scene3_lr00001000_struct1000_200_400_eps5000_NotOpt.h5'

        main(params)

    # Bayesian Optimization
    else:
        bayesOpt = BayesianOptimizer(params)
        bayesOpt.optimize_raytracer()




