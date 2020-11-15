"""
Class of the Bayesian Optimizer
"""
from GPyOpt.methods import BayesianOptimization
from py_smallpt import main

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

            self.params['img_title'] = 'DDQN_scene{}_lr{}_struct{}-{}_{}-{}_{}-{}_eps{}'.format(self.params['scene'],
                                                                                        lr_string,
                                                                                        self.params['dense_layer12'],
                                                                                        self.params['dense_layer3'],
                                                                                        self.params['state_layer1'],
                                                                                        self.params['state_layer2'],
                                                                                        self.params['advantage_layer1'],
                                                                                        self.params['advantage_layer2'],
                                                                                        self.params['epsilon_decay_linear'])
            self.params['weight'] = 'weights_scene{}_'.format(self.params['scene']) + self.params['img_title'] + '.h5'
            self.params['training'] = True
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

        with open(self.params['path_SSIM_total'], 'a') as file:
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