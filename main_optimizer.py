import os

from optimization.Gradient_descent_optimisation import two_layer_optimization

if __name__ == "__main__":
    os.environ["df_stage_name"] = "df_stage_barchart_1m"
    os.environ["optima_metric"] = "total"

    # adaptive learning rate adjustment: epsilon
    # set large to get better results because of faster convergence to escape local minima
    os.environ["epsilon_growth_factor"] = '0.2'  

    # Use central difference for gradient approximation
    # pros: more accurate, cons: more expensive
    os.environ['use_central_difference'] = 'True'

    # learning rate strategy for gradient descent DEFAULT is decay
    os.environ["lr_schedule"] = '5'
    os.environ["lr_schedule_factor"] = '0.1'

    os.environ["lr_adaptive_improvement_threshold"] = '0.1'
    os.environ["lr_adaptive_increase_factor"] = '1.1'
    os.environ["lr_adaptive_decrease_factor"] = '2'

    two_layer_optimization(batch_params_file_name = "barchart_1m",
                            patience = 5,
                            lr_adaptive_strategy = True,
                            metric_min_delta = 0.1,
                            worker_count=14, # 27 params to optimise, but i only have 16 cores, so how do i decide: 16, 8, 4, 2, 1?
                            gradient_descent_max_iterations =10,
                            global_minimum = True, 
                            # This set better_param set as initial_param_set for further ---> Could be overfitting if True
                            )


