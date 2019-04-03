from IPython.display import display, clear_output
import matplotlib.pylab as plt
import numpy as np

def plot_learning(plot_handles, ylabel):
    plt.legend(handles=plot_handles, loc="center right")
    plt.xlabel("Iterations")
    plt.ylabel(ylabel)
    
def plot_training(log_every, train_loss, val_loss, title, ylabel):
    clear_output(wait=False)
    plt.gcf().clear()
    iters = np.arange(0,len(train_loss))*log_every
    iters_val = np.arange(0,len(val_loss))*log_every
    #train_plot, = plt.plot(iters, train_loss, 'r', label="training")
    val_plot, = plt.plot(iters, val_loss, 'b', label="validation")
    
    plot_learning([val_plot], ylabel)
    plt.title(title)

    display(plt.gcf())