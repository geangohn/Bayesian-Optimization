import numpy as np

import matplotlib.pyplot as plt
import GPy
import GPyOpt
from GPyOpt.util import mcmc_sampler

from matplotlib import pyplot as plt

def plot_one_dimensional_function(function_call, uniform_design, selected_design, 
                                  axis_bounds=[0, 1, -6.5, 16.5]):
    """ Plot one dimensional function at given points

    Inputs
    -------
    function_call : an object 
        an object with method f that calculates the target function
    uniform_design : 1d np.array
        uniform design of points to plot solid line
    selected_design : 1d np.array  
        selected design to plot as dots
    axis_bounds : list
        list of length 4 that defines axis bounds. Default values correspond to
        Forrester function
    """
    plt.figure(figsize=(8, 7))
    plt.plot(uniform_design, function_call.f(uniform_design), 
             label='Forrester function')
    plt.scatter(selected_design, function_call.f(selected_design), s=50, 
             label='Initial sample')

    plt.xlabel('x', fontsize=22)
    plt.ylabel('f(x)', fontsize=22)
    plt.legend(fontsize=18, loc='upper left')
    plt.axis(axis_bounds);

    
def run_bayesian_optimization(acquisition_function_name='EI', beta=2):
    u""" Run Bayesian optimization for a given acquisition function

    Inputs
    --------
    acquisition_function_name : string 
        either EI, UCB or PI
    beta : positive float
        beta for UCB acquisition function. Don't used by other acquisition
        functions

    Returns  
    --------
    targets_history : list of floats
      values of targets during optimization
    """
    forrester_function = GPyOpt.objective_examples.experiments1d.forrester()

    space = [{'name': 'x', 'type': 'continuous', 'domain': (0, 1)}]
    design_region = GPyOpt.Design_space(space=space)

    initial_sample_size = 5
    initial_design = GPyOpt.experiment_design.initial_design(
            'random', design_region, initial_sample_size)

    # The target function
    objective = GPyOpt.core.task.SingleObjective(forrester_function.f)

    # Model type
    gp_model = GPyOpt.models.GPModel(exact_feval=True, optimize_restarts=10,
                                     verbose=False)
    # exact_feval - are evaluations exact?
    # optimize_restarts - number of restarts at each step
    # verbose - how verbose we are

    # Optimizer of the acquisition function, the default is 'LBFGS'
    aquisition_optimizer = GPyOpt.optimization.AcquisitionOptimizer(
            design_region)

    # The selected acquisition function
    if acquisition_function_name == 'EI': #  Expected improvement
        acquisition_function = GPyOpt.acquisitions.AcquisitionEI(
                gp_model, design_region, optimizer=aquisition_optimizer)
    elif acquisition_function_name == 'UCB': # Upper confidence bound 
        acquisition_function = GPyOpt.acquisitions.AcquisitionLCB(
                gp_model, design_region, optimizer=aquisition_optimizer,
                exploration_weight=beta)
    elif acquisition_function_name == 'PI': # Probability of improvement
        acquisition_function = GPyOpt.acquisitions.AcquisitionMPI(
                gp_model, design_region, optimizer=aquisition_optimizer)
    elif acquisition_function_name == 'ES': # Entropy search     
        sampler = mcmc_sampler.AffineInvariantEnsembleSampler(design_region)
        acquisition_function = GPyOpt.acquisitions.AcquisitionEntropySearch(
                gp_model, design_region, optimizer=aquisition_optimizer,
                sampler=sampler)
    else:
        raise ValueError(u"Invalid name for acquisition_function_name. "
                         u"Possible values are 'EI', 'UCB', 'PI', 'ES' ")

    # How we collect the data
    evaluator = GPyOpt.core.evaluators.Sequential(acquisition_function)

    # Now we are ready to construct the procedure
    bayesian_optimizer = GPyOpt.methods.ModularBayesianOptimization(
            gp_model, design_region, objective, acquisition_function,
            evaluator, initial_design)

    # Run five iterations
    max_iter = 10
    max_time = None
    tolerance = 1e-8
    bayesian_optimizer.run_optimization(max_iter=max_iter, max_time=max_time, 
                                        eps=tolerance, verbosity=False) 

    return bayesian_optimizer.Y_best


def plot_acquisition(optimizer):
    
    x_grid = np.linspace(0, 1, 200)
    acqu = optimizer.acquisition.acquisition_function(x_grid.reshape(-1, 1))

    y_mean, y_var = optimizer.model.model.predict(x_grid.reshape(-1, 1))
    y_std = np.sqrt(y_var)
    
    suggested_sample = optimizer.suggest_next_locations()
    

    if max(-acqu - min(-acqu)) > 0:
        acqu_normalized = (-acqu - min(-acqu))/(max(-acqu - min(-acqu)))
    else:
        acqu_normalized = (-acqu - min(-acqu))

    y_norm = optimizer.Y
    y_norm = y_norm - y_norm.mean()
    if y_norm.std() > 0:
        y_norm /= y_norm.std()

    factor = max(y_mean + 1.96 * y_std) - min(y_mean - 1.96 * y_std)
    y_lower = y_mean - 1.96 * y_std
    y_upper = y_mean + 1.96 * y_std

    plt.plot(optimizer.X, y_norm, '.r', markersize=10)
    
    plt.plot(x_grid, 0.2 * factor * acqu_normalized
                     - abs(min(y_mean - 1.96 * y_std)) - 0.25 * factor,
             '-r', lw=2, label='Acquisition')
    
    plt.plot(x_grid, y_mean, '-k', lw=1, alpha=0.6)
    plt.plot(x_grid, y_upper, '-k', alpha=0.2)
    plt.plot(x_grid, y_lower, '-k', alpha=0.2)
    
    color = plt.rcParams['axes.prop_cycle'].by_key()['color'][0]
    plt.fill_between(x_grid, y_lower.ravel(), y_upper.ravel(), color=color,
                     alpha=0.1)
    
    plt.ylim(min(y_mean - 1.96 * y_std) - 0.25 * factor,
             max(y_mean + 1.96 * y_std) + 0.05 * factor)
    plt.axvline(x=suggested_sample[len(suggested_sample)-1], color='r')
    plt.xlabel('x', fontsize=14)
    plt.ylabel('f(x)', fontsize=14)
    plt.legend()
    plt.show()