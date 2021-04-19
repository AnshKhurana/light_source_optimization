import numpy as np
from matplotlib import pyplot as plt
import pickle as pkl

def save_plot(values, x, title, xtitle, ytitle, save_name):

    plt.plot(x, values, marker='^')

    # plt.xscale("log")
    plt.legend()
    plt.xticks(x, x)
    plt.title(title)
    plt.xlabel(xtitle)
    plt.ylabel(ytitle)
    plt.savefig(save_name)
    plt.close()

def save_combined(val_ours, val_scipy, x, title, xtitle, ytitle, save_name):

    plt.plot(x, val_ours, label='Ours', marker='^')
    plt.plot(x, val_scipy, label='Scipy', marker = '^')
    # plt.xscale("log")
    plt.legend()
    plt.xticks(x, x)
    plt.title(title)
    plt.xlabel(xtitle)
    plt.ylabel(ytitle)
    plt.savefig(save_name)
    plt.close()


num_bulbs = 4
x = list(range(1, 11))
# [SciPy_nm, SciPy_cg, FR, SciPy_nm_nit, SciPy_cg_nit, FR_nit, SciPy_nm_time, SciPy_cg_time, FR_time] = pkl.load( open('5bulbs_25-35.pkl','rb'))
el1 = pkl.load( open('data/nelder_mead/4bulbs.pkl','rb'))
# el2 = pkl.load( open('6bulbs_50-55.pkl','rb'))

[SciPy_cg, FR, SciPy_cg_nit, FR_nit, SciPy_cg_time, FR_time] = el1

# print(len(SciPy_cg))

def plot_values():
    save_plot(SciPy_cg, x, 'Minimum Values Scipy CG (%d bulbs)' % num_bulbs, 'Seeds', 'Obj Value', 'scipy_cg_%d.png' % num_bulbs)
    print('Mean obj: ', np.mean(SciPy_cg), 'Std obj: ', np.std(SciPy_cg))
    save_plot(FR, x, 'Minimum Values our FR (%d bulbs)' % num_bulbs, 'Seeds', 'Obj Value', 'ours_fr_%d.png' % num_bulbs)
    print('Mean obj: ', np.mean(FR), 'Std obj: ', np.std(FR))
    save_combined(FR, SciPy_cg, x, 'Optimal Obj Values (%d bulbs)' % num_bulbs, 'Seeds', 'Obj Value', 'comparison_value_%d.png' % num_bulbs)

def plot_runtime():
    print('Mean runtime: ', np.mean(SciPy_cg_time), 'Std obj: ', np.std(SciPy_cg_time))
    print('Mean runtime ours: ', np.mean(FR_time), 'Std runtime ours: ', np.std(FR_time))

    save_plot(FR_time, x, 'Ours Runtime (%d bulbs)' % num_bulbs, 'Seeds', 'Runtime (s)', 'ours_time_%d.png' % num_bulbs)
    save_plot(SciPy_cg_time, x, 'Scipy Runtime (%d bulbs)' % num_bulbs, 'Seeds', 'Runtime (s)', 'scipy_time_%d.png' % num_bulbs)

    save_combined(FR_time, SciPy_cg_time, x, 'Algorithm Runtime (%d bulbs)' % num_bulbs, 'Seeds', 'Runtime (s)', 'comparison_time_%d.png' % num_bulbs)


def plot_iter():
    print('Mean iter: ', np.mean(SciPy_cg_nit), 'Std obj: ', np.std(SciPy_cg_nit))
    print('Mean iter ours: ', np.mean(FR_nit), 'Std iter ours: ', np.std(FR_nit))

    save_plot(FR_nit, x, 'Ours Iterations till convergence (%d bulbs)' % num_bulbs, 'Seeds', '#iterations', 'ours_iter_%d.png' % num_bulbs)
    save_plot(SciPy_cg_nit, x, 'Scipy Iterations till convergence (%d bulbs)' % num_bulbs, 'Seeds', '#iterations', 'scipy_iter_%d.png' % num_bulbs)

    save_combined(FR_nit, SciPy_cg_nit, x, 'Algorithm #Iterations (%d bulbs)' % num_bulbs, 'Seeds', '#iterations', 'comparison_iter_%d.png' % num_bulbs)



def plot_niter_variation():
    FR = [0.009803663939237595, 0.009804542176425457, 0.009775332175195217, 0.0098054064437747, 0.009790563024580479, 0.009763374924659729]
    FR_time = [85.80639266967773, 125.18251419067383, 39.19958305358887, 323.34411692619324, 355.57732462882996, 270.8850338459015]
    x = list(range(1, 7))

    save_plot(FR, x, 'Objective Variation with n_iter', 'n_iter', 'Min Obj value', 'obj_niter.png')
    save_plot(FR_time, x, 'Runtime Variation with n_iter', 'n_iter', 'Runtime (s.)',
              'runtime_niter.png')

def plot_gamma_variation():
    Scipy_obj = [-0.4399, -0.20, 0.094]
    Ours_obj = [-0.1, 0.098]
    pass

if __name__ == '__main__':
    plot_runtime()