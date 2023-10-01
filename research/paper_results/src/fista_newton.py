import adelie as gl
import matplotlib.pyplot as plt
import timeit
import numpy as np


def check_acc(beta, L, v, l1, l2):
    return beta - (v /(L + l2 + l1 / np.sqrt(np.sum(beta**2))))


def bench(ps, configs):
    algs_dict = {
        'Newton': gl.newton_solver,
        'Newton-ABS': gl.newton_abs_solver,
        'Newton-Brent': gl.newton_brent_solver,
        'Brent': gl.brent_bu,
        'ISTA': gl.ista_solver,
        'FISTA': gl.fista_solver,
        'FISTA-ADA': gl.fista_adares_solver,
    }

    index_names = list(algs_dict.keys())
    if 'skip' in configs:
        index_names = [ind for ind in index_names if ind not in configs['skip']]

    seed = configs['seed']
    min_eval = configs['min_eval']
    max_eval = configs['max_eval']
    l1 = configs['l1']
    l2 = configs['l2']
    newton_tol = configs['newton_tol']
    fista_tol = configs['fista_tol']
    max_iters = configs['max_iters']
    newton_times = configs['newton_times']
    fista_times = configs['fista_times']
    
    tols_dict = {
        'Newton': newton_tol,
        'Newton-ABS': newton_tol,
        'Newton-Brent': newton_tol,
        'Brent': newton_tol, 
        'ISTA': fista_tol,
        'FISTA': fista_tol,
        'FISTA-ADA': fista_tol,
    }

    numbers_dict = {
        'Newton': newton_times,     
        'Newton-ABS': newton_times,
        'Newton-Brent': newton_times,
        'Brent': newton_times,
        'ISTA': fista_times,
        'FISTA': fista_times,
        'FISTA-ADA': fista_times,
    }

    n_ps = len(ps)
    n_algs = len(index_names)
    times = np.zeros((n_algs, n_ps))
    iters = np.zeros((n_algs, n_ps))
    accs = np.zeros((n_algs, n_ps))
    data = []
    
    for i, p in enumerate(ps):
        np.random.seed((seed + 100*i) % 1000007)
        
        # generate data
        L_zero_prop = configs['L_zero_prop']
        L_small_prop = configs['L_small_prop']
        L, v = gl.data.generate_block_norm_data(
            p, L_zero_prop, L_small_prop, large_min=min_eval, large_max=max_eval,
        )

        # save data
        data.append((L, v))

        def _bench(ind):
            tm = timeit.repeat(
                lambda: algs_dict[ind](L, v, l1, l2, tols_dict[ind], max_iters), 
                repeat=1, 
                number=numbers_dict[ind],
            )[0] / numbers_dict[ind]
            return tm

        # benchmark
        curr_times = [_bench(ind) for ind in index_names]

        # call for outputs
        curr_outs = [
            algs_dict[ind](L, v, l1, l2, tols_dict[ind], max_iters)
            for ind in index_names
        ]
        
        # save output
        times[:, i] = curr_times
        iters[:, i] = [out['iters'] for out in curr_outs]
        accs[:, i] =[
            np.max(np.abs(check_acc(out['beta'], L, v, l1, l2)))
            for out in curr_outs
        ]

    return {
        'ps': ps, 
        'configs': configs, 
        'data': data, 
        'index_names': index_names, 
        'times': times, 
        'iters': iters, 
        'accs': accs,
    }


def bench_assess(bench_out, filename=None):
    ps = bench_out['ps']
    index_names = bench_out['index_names']
    times = bench_out['times']
    iters = bench_out['iters']
    accs = bench_out['accs']

    markers = ['.', '*', 'v', 'x', '1', '2']
    linestyles = ['-.'] * len(markers)

    
    if 'Newton-ABS' in index_names:
        best_pos = index_names.index('Newton-ABS')
        rel_times = times / times[best_pos][None]
        types = [times, rel_times, iters, accs]
        type_names = ["Time", "Relative Time", "Newton Iterations", "Accuracy"]
        ylabels = ["Time (s)", "Relative Time to Newton-ABS", "Iterations", "$\max_i |\hat{\\beta}_i - \\beta^{\star}_i |$"]

        fig, axes = plt.subplots(2, 2, figsize=(8, 8))
        for row in range(2):
            for col in range(2):
                j = 2 * row + col
                ax = axes[row, col]
                vals = types[j]
                for i, vals_alg in enumerate(vals):
                    ax.plot(ps, vals_alg, marker=markers[i], label=index_names[i], linestyle=linestyles[i])
                ax.legend()
                ax.set_yscale('log')
                ax.set_xscale('log')
                ax.set_title(f"{type_names[j]} Comparison")
                ax.set_xlabel("Number of features")
                ax.set_ylabel(ylabels[j])
                
    else:
        types = [iters, times, accs]
        type_names = ["Newton Iterations", "Time", "Accuracy"]
        ylabels = ["Iterations", "Time (s)", "$\max_i |\hat{\\beta}_i - \\beta^{\star}_i |$"]

        fig, axes = plt.subplot_mosaic(
            [['upper left', 'right'], 
             ['lower left', 'right']],
            figsize=(8, 8),
        )
        for j, k in enumerate(axes):
            ax = axes[k]
            vals = types[j]
            for i, vals_alg in enumerate(vals):
                ax.plot(ps, vals_alg, marker=markers[i], label=index_names[i], linestyle=linestyles[i])
            ax.legend()
            ax.set_yscale('log')
            ax.set_xscale('log')
            ax.set_title(f"{type_names[j]} Comparison")
            ax.set_xlabel("Number of features")
            ax.set_ylabel(ylabels[j])
        
    plt.tight_layout()

    bench_out['plot'] = fig

    if not (filename is None):
        plt.savefig(f"figures/pgd_newton_{filename}.pdf", bbox_inches='tight')
    plt.show()

    return bench_out