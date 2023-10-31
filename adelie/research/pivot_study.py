import adelie as ad
import numpy as np
import array
import csv
from sklearn.preprocessing import SplineTransformer
from scipy.sparse import csr_matrix
from time import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
from adelie.logger import logger
from IPython.display import HTML


root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
tex_path = os.path.join(root_path, "docs/tex/pivot_rule")
data_path = os.path.join(root_path, "data")
fig_path = os.path.join(tex_path, "figures")


def active_sets(state):
    p = state.X.cols()
    betas = state.betas
    feature_to_group = np.empty(p, dtype=int)
    curr_idx = 0
    for i, gs in enumerate(state.group_sizes):
        feature_to_group[curr_idx:curr_idx+gs] = i
        curr_idx += gs
    active_sets = [
        set(feature_to_group[betas[i].indices])
        for i in range(betas.shape[0])
    ]
    return active_sets


def _screen_sets_strong(
    state,
    *,
    scores: np.ndarray,
    active_sets: list,
    safe_sets: list =None,
    method: str ="",
):
    n_lmdas, G = scores.shape
    lmdas = state.lmdas

    strong_threshs = (
        2 * lmdas[1:]  - lmdas[:-1]
    )
    is_strong = scores[:-1] >= strong_threshs[:, None]

    predict_sets = []
    gns = np.arange(G)
    for i in range(n_lmdas-1):
        predict_set = set(gns[is_strong[i]])
        
        # always add current active set
        predict_set = predict_set.union(active_sets[i])

        if "ever" in method and i > 0:
            predict_set = predict_set.union(predict_sets[i-1])

        if "safe" in method:
            predict_set = predict_set.intersection(safe_sets[i])

        if "lazy" in method:
            previous_predict_set = predict_set if i == 0 else predict_sets[i-1]
            if active_sets[i].issubset(previous_predict_set):
                predict_set = previous_predict_set

        if not active_sets[i+1].issubset(predict_set):
            logger.warning("Strong rule failed!")

        predict_sets.append(predict_set)

    return predict_sets


def _screen_sets_safe(
    state,
    *,
    resids: np.ndarray,
    active_sets: list,
    method: str ="",
):
    if not (state.setup_edpp and state.use_edpp):
        raise RuntimeError("EDPP quantities are not properly set up! Run the solver with EDPP enabled.")
    X = state.X
    X_group_norms = state.X_group_norms
    penalty = state.penalty
    betas = state.betas
    edpp_v1_0 = state.edpp_v1_0
    edpp_resid_0 = state.edpp_resid_0
    lmdas = state.lmdas
    G, n_lmdas, n, p = X_group_norms.shape[0], betas.shape[0], X.rows(), X.cols()

    v1s = np.empty((n_lmdas-1, n))
    v1s[0] = edpp_v1_0
    v1s[1:] = (edpp_resid_0[None] - resids[1:-1]) / lmdas[1:-1][:, None]
    v2s = edpp_resid_0[None] / lmdas[1:][:, None] - resids[:-1] / lmdas[:-1][:, None]
    v2_perps = v2s - (np.sum(v1s * v2s, axis=-1) / np.sum(v1s ** 2, axis=-1))[:, None] * v1s
    v2_perp_norms = np.linalg.norm(v2_perps, axis=-1)
    edpps = np.empty((n_lmdas-1, p))
    for i in range(n_lmdas-1):
        X.bmul(0, p, (resids[i] / lmdas[i] + 0.5 * v2_perps[i]), edpps[i])
    abs_edpps = ad.diagnostic.gradient_norms(state, grads=edpps)
    is_edpp = (
        abs_edpps >= (penalty[None] - 0.5 * v2_perp_norms[:, None] * X_group_norms[None])
    )

    predict_sets = []
    gns = np.arange(G)
    for i in range(n_lmdas-1):
        predict_set = set(gns[is_edpp[i]])

        # always add current active set
        predict_set = predict_set.union(active_sets[i])

        if ("ever" in method) and i > 0:
            predict_set = predict_set.union(predict_sets[i-1])

        if ("ever" in method) and (not active_sets[i+1].issubset(predict_set)):
            logger.critical(
                "EDPP ever safe set does not contain the next active set!"
            )

        predict_sets.append(predict_set)

    return predict_sets


def _screen_sets_pivot(
    state,
    *,
    scores: np.ndarray,
    active_sets: list,
    safe_sets: list =None,
    method: str ="",
):
    betas = state.betas
    n_lmdas, G = scores.shape

    orders = np.argsort(scores, axis=-1)
    K = betas[0].count_nonzero()

    def _fixed_update(
        predict_set,
        slack_set,
        delta,
    ):
        slack_set = np.array(list(slack_set - predict_set), dtype=int)
        slack_scores = scores[i, slack_set]
        slack_scores_order = np.argsort(slack_scores)
        delta_size = min(delta, len(slack_scores_order))
        return predict_set.union(
            set(slack_set[slack_scores_order[-delta_size:]])
        )

    ever_active_sets = []
    for i, s in enumerate(active_sets):
        if i == 0:
            ever_active_sets.append(s)
        else:
            ever_active_sets.append(
                ever_active_sets[-1].union(s)
            )

    predict_sets = []
    for i in range(n_lmdas-1):
        subset_size = int(min(
            max(int(K * (1 + state.pivot_subset_ratio)), state.pivot_subset_min),
            G,
        ))
        order_sub = orders[i, -subset_size:]
        argmin, _ = ad.optimization.search_pivot(
            np.arange(order_sub.shape[0]), 
            scores[i, order_sub],
        )
        argmin = int(argmin * (1 - state.pivot_slack_ratio))
        cutoff_idx = argmin + G - len(order_sub)
        predict_set = set(orders[i, cutoff_idx:])

        # always add current active set
        predict_set = predict_set.union(active_sets[i])

        if "safe" in method:
            predict_set = predict_set.intersection(safe_sets[i])

        if "ever" in method and i > 0:
            predict_set = predict_set.union(predict_sets[i-1])

        if "lazy" in method:
            pivot_predict_set = predict_set
            previous_predict_set = predict_set if i == 0 else predict_sets[i-1]

            # if KKT passed from using previous predict set, 
            # add however much active set increased.
            # This small hedging is used for next lambda.
            if active_sets[i].issubset(previous_predict_set):
                delta_active = max(
                    max(len(ever_active_sets[i]) - len(ever_active_sets[i-1]), 0) if i > 0 else 0,
                    0,
                )

                if delta_active == 0:
                    predict_set = previous_predict_set
                else:
                    if "safe" in method:
                        slack_set = safe_sets[i]
                    else:
                        slack_set = set(np.arange(G))
                    predict_set = _fixed_update(
                        previous_predict_set,
                        slack_set,
                        delta_active,
                    )

                # if KKT failed after this small hedging,
                # use pivot-rule instead.
                if not active_sets[i+1].issubset(predict_set):
                    logger.warning(f"Index {i}: small hedge failed! Using pivot.")
                    predict_set = pivot_predict_set

            # if KKT failed, use pivot-rule.
            else:
                logger.warning(f"Index {i}: KKT failed! Using pivot.")
                predict_set = pivot_predict_set

        if "greedy" in method:
            # if current prediction does not contain the next active set,
            # fallback to greedy method
            count = 0
            while not active_sets[i+1].issubset(predict_set):
                if "safe" in method:
                    slack_set = safe_sets[i]
                else:
                    slack_set = set(np.arange(G))
                predict_set = _fixed_update(
                    predict_set,
                    slack_set,
                    state.delta_strong_size,
                )
                count += 1
            if count >= 1:
                logger.warning(f"Index {i}: Greedy occured {count} times!")

        if not active_sets[i+1].issubset(predict_set):
            logger.warning("Pivot rule failed!")

        K = len(predict_set)
        predict_sets.append(predict_set)

    return predict_sets


def screen_sets(
    state, 
    *,
    screen_rule: str ="pivot",
    **kwargs,
):
    dct = {
        "strong": _screen_sets_strong,
        "safe": _screen_sets_safe,
        "pivot": _screen_sets_pivot,
    }
    return dct[screen_rule](state, **kwargs)


def plot_scores(
    state,
    *,
    scores: np.ndarray,
    active_sets: list,
    sorted: bool =False,
    idx: int =None,
    add_active_color: bool =True,
    add_cutoff: bool =True,
):
    n_lmdas, G = scores.shape

    do_anim = idx is None
    idx = 0 if do_anim else idx

    gns = np.arange(G)

    colors = ["black", "red"]
    labels = ["never-active", "ever-active"]
    alphas = [0.6, 0.8]

    if not add_active_color:
        colors[1] = colors[0]
        labels = [None] * 2
        alphas[1] = alphas[0]

    fig, ax = plt.subplots(layout="constrained")

    order = np.argsort(scores[idx])
    order_sub = order[-int((1 + state.pivot_subset_ratio) * state.strong_sizes[idx]):]
    argmin, _ = ad.optimization.search_pivot(
        np.arange(order_sub.shape[0]), 
        scores[idx, order_sub]
    )
    argmin = int(argmin * (1 - state.pivot_slack_ratio))
    cutoff_idx = argmin + G - len(order_sub)

    is_active = np.zeros(G, dtype=bool)
    is_active[np.array(list(active_sets[idx+1]))] = True
    if sorted:
        xs = [
            gns[:cutoff_idx],
            gns[cutoff_idx:],
        ]
        ys = [
            scores[idx, order[:cutoff_idx]],
            scores[idx, order[cutoff_idx:]],
        ]
    else:
        xs = [
            gns[~is_active],
            gns[is_active],
        ]
        ys = [
            scores[idx, ~is_active],
            scores[idx, is_active],
        ]
    scats = [None] * 2
    for i, (x, y, color, label, alpha) in enumerate(zip(xs, ys, colors, labels, alphas)):
        scats[i] = ax.scatter(
            x, y,
            color=color,
            marker=".",
            facecolor="None",
            alpha=alpha,
            label=label,
        )

    if add_cutoff:
        line = ax.axhline(
            scores[idx, order[cutoff_idx]], 
            linestyle='--', 
            linewidth=1, 
            color='green',
            label=f"{state.screen_rule}-cutoff",
        )
    legend = ax.legend()
    # if there's nothing in the legend, just remove it
    if len(legend.get_texts()) == 0:
        legend.remove()
    ax.set_title("Active Scores")
    ax.set_ylabel("Active Scores")
    ax.set_xlabel("Group Number" + (" (sorted)" if sorted == True else ""))

    if do_anim:
        plt.close(fig)
    else:
        return fig, ax

    def update(idx):
        s = scores[idx]

        order = np.argsort(s) 
        order_sub = order[-int((1 + state.pivot_subset_ratio) * state.strong_sizes[idx]):]
        argmin, _ = ad.optimization.search_pivot(
            np.arange(order_sub.shape[0]), 
            s[order_sub]
        )
        argmin = int(argmin * (1 - state.pivot_slack_ratio))
        cutoff_idx = argmin + G - len(order_sub)

        is_active = np.zeros(G, dtype=bool)
        is_active[np.array(list(active_sets[idx+1]))] = True
        if sorted:
            xs = [
                gns[:cutoff_idx],
                gns[cutoff_idx:],
            ]
            ys = [
                s[order[:cutoff_idx]],
                s[order[cutoff_idx:]],
            ]
        else:
            xs = [
                gns[~is_active],
                gns[is_active],
            ]
            ys = [
                s[~is_active],
                s[is_active],
            ]
        for i, (x, y, color, label, alpha) in enumerate(zip(xs, ys, colors, labels, alphas)):
            data = np.stack([x, y]).T
            scats[i].set_offsets(data)
            scats[i].set(
                color=color,
                facecolor="None",
                alpha=alpha,
                label=label,
            )
        ax.set(ylim=[0, np.max(s) * 1.05])
        line.set_ydata(s[order[cutoff_idx]])
        return (scats[0], scats[1],)

    anim = animation.FuncAnimation(
        fig=fig, 
        func=update, 
        frames=n_lmdas-1, 
        interval=200, 
        repeat=False,
    )

    return HTML(anim.to_html5_video())


def plot_set_sizes(
    state,
    *,
    mapping: dict,
    ax=None,
):
    lmdas = state.lmdas

    dct = {
        "EDPP": 0,
        "strong": 1,
        "pivot-S": 2,
        "pivot-SL": 3,
        "active": 4,
    } 
    markers = ["o", "^", "v", "*", "."]
    colors = ["tab:green", "tab:purple", "tab:orange", "tab:blue", "tab:red"]

    make_ax = ax is None
    if make_ax:
        fig, ax = plt.subplots(layout="constrained")

    tls = -np.log(lmdas[1:])

    for k, v in mapping.items():
        idx = dct[k]
        ax.plot(
            tls, 
            v, 
            linestyle="None",
            color=colors[idx],
            marker=markers[idx],
            markerfacecolor="None",
            label=k,
        )
    ax.legend()
    ax.set_title("Set Size Comparison")
    ax.set_ylabel("Number of Groups")
    ax.set_xlabel(r"$-\log(\lambda)$")

    if make_ax: 
        return fig, ax
    return ax


def arcene(path):
    X_train = np.genfromtxt(
        os.path.join(path, "arcene_train.data"),
    )
    y_train = np.genfromtxt(
        os.path.join(path, "arcene_train.labels")
    )

    subset = np.std(X_train, axis=0) > 0
    X_train = X_train[:, subset]

    n, _ = X_train.shape
    X_train /= np.std(X_train, axis=0)[None] * np.sqrt(n)
    X_train = np.asfortranarray(X_train)
    y_train /= np.sqrt(n)

    return X_train, y_train
    

def dorothea(path):
    def csv_to_csr(f, p, delimiter=" "):
        """Read content of CSV file f, return as CSR matrix."""
        data = array.array("f")
        indices = array.array("i")
        indptr = array.array("i", [0])
    
        for i, row in enumerate(csv.reader(f, delimiter=delimiter), 1):
            row = np.array([int(r) for r in row[:-1]])
            data.extend(np.ones(len(row)))
            indices.extend(row)
            indptr.append(indptr[-1]+len(row))
    
        return csr_matrix(
            (data, indices, indptr),
            dtype=float, shape=(i, p)
        )

    X_train = csv_to_csr(
        open(os.path.join(path, "dorothea_train.data")),
        p=100000,
    )
    X_train = X_train.toarray()
    y_train = np.genfromtxt(
        os.path.join(path, "dorothea_train.labels")
    )

    subset = np.std(X_train, axis=0) > 0
    X_train = X_train[:, subset]

    n, _ = X_train.shape
    X_train /= np.std(X_train, axis=0)[None] * np.sqrt(n)
    X_train = np.asfortranarray(X_train)
    y_train /= np.sqrt(n)

    return X_train, y_train


def gisette(path):
    X_train = np.genfromtxt(
        os.path.join(path, "gisette_train.data"),
    )
    y_train = np.genfromtxt(
        os.path.join(path, "gisette_train.labels")
    )

    subset = np.std(X_train, axis=0) > 0
    X_train = X_train[:, subset]

    n, _ = X_train.shape
    X_train /= np.std(X_train, axis=0)[None] * np.sqrt(n)
    X_train = np.asfortranarray(X_train)
    y_train /= np.sqrt(n)

    return X_train, y_train


def mnist(path):
    data = np.genfromtxt(
        os.path.join(path, "mnist_train.csv"),
        delimiter=",",
    )
    X_train = data[1:-1, 1:].T
    y_train = data[-1, 1:]

    subset = np.std(X_train, axis=0) > 0
    X_train = X_train[:, subset]

    n, _ = X_train.shape
    X_train /= np.std(X_train, axis=0)[None] * np.sqrt(n)
    X_train = np.asfortranarray(X_train)
    y_train /= np.sqrt(n)

    X_train.shape, y_train.shape

    return X_train, y_train


def electricity(path):
    def conv(x):
        return x.replace(",", ".").encode()

    data = np.genfromtxt(
        (conv(x) for x in open(os.path.join(path, "LD2011_2014.txt"))),
        skip_header=1,
        delimiter=";",
    )
    X_train = data[:-1, 1:].T
    y_train = data[-1, 1:].flatten()

    subset = np.std(X_train, axis=0) > 0
    X_train = X_train[:, subset]

    n, _ = X_train.shape
    X_train /= np.std(X_train, axis=0)[None] * np.sqrt(n)
    X_train = np.asfortranarray(X_train)
    y_train /= np.sqrt(n)

    return X_train, y_train


def gene(path):
    X_train = np.genfromtxt(
        os.path.join(path, "data.csv"),
        skip_header=1,
        delimiter=",",
    )
    y_dict = {
        b"PRAD": 0,
        b"LUAD": 1,
        b"BRCA": 2,
        b"KIRC": 3,
        b"COAD": 4,
    }
    y_train = np.genfromtxt(
        os.path.join(path, "labels.csv"),
        skip_header=1,
        delimiter=",",
        converters={
            1 : lambda x: y_dict[x],
        },
    )

    X_train = X_train[:, 1:]
    y_train = np.array([y[1] for y in y_train], dtype=float)

    subset = np.std(X_train, axis=0) > 0
    X_train = X_train[:, subset]

    n, _ = X_train.shape
    X_train /= np.std(X_train, axis=0)[None] * np.sqrt(n)
    X_train = np.asfortranarray(X_train)
    y_train /= np.sqrt(n)

    return X_train, y_train


def spline_basis(X, **kwargs):
    spl_tr = SplineTransformer(
        include_bias=False,
        order="F",
        **kwargs,
    )
    n = X.shape[0]
    X = X * np.sqrt(n)
    X = spl_tr.fit_transform(X)
    X /= np.sqrt(n)
    return X


def real_data_analysis(
    X: np.ndarray,
    y: np.ndarray,
    configs: dict,
):
    start = time()
    strong_state = ad.grpnet(
        X=X,
        y=y,
        **configs,
        screen_rule="strong",
        lazify_screen=False,
    )
    end = time()
    strong_time = end - start

    start = time()
    pivot_state = ad.grpnet(
        X=X,
        y=y,
        **configs,
        screen_rule="pivot",
        lazify_screen=True,
    )
    end = time()
    pivot_time = end - start

    active_ss = active_sets(pivot_state)
    active_sizes = np.array([len(s) for s in active_ss])
    strong_sizes = strong_state.strong_sizes
    pivot_sizes = pivot_state.strong_sizes

    assert len(strong_sizes) == len(pivot_sizes)

    labels = ["strong", "pivot-SL", "active"]
    markers = ["v", "*", "."]
    colors = ["tab:orange", "tab:blue", "tab:red"]
    ys = [
        strong_sizes,
        pivot_sizes,
        active_sizes,
    ]
    
    def _kkt_fails(ns):
        count = 0
        out = []
        for n in ns:
            if n == 1:
                out.append(count)
                count = 0
                continue
            count += 1
        return np.array(out, dtype=int)

    kkt_fails = [
        _kkt_fails(strong_state.n_valid_solutions),
        _kkt_fails(pivot_state.n_valid_solutions),
        np.zeros(len(pivot_sizes), dtype=int),
    ]

    fig, ax = plt.subplots(layout="constrained")
    tls = -np.log(pivot_state.lmdas)

    for y, kkt, label, marker, color in zip(
        ys, kkt_fails, labels, markers, colors
    ):
        # plot KKT successful ones
        ax.plot(
            tls[kkt == 0], 
            y[kkt == 0], 
            linestyle="None",
            color=color,
            marker=marker,
            label=label,
            markerfacecolor="None",
        )
        # plot KKT failure ones
        ax.plot(
            tls[kkt > 0],
            y[kkt > 0],
            linestyle="None",
            color=color,
            marker="x",
            label=None,
            markerfacecolor="None",
        )
    ax.legend()
    ax.set_title("Set Size Comparison")
    ax.set_ylabel("Number of Groups")
    ax.set_xlabel(r"$-\log(\lambda)$")

    return (
        fig, 
        ax,
        X.shape[0],
        pivot_state.groups.shape[0],
        strong_state,
        pivot_state,
        strong_time,
        pivot_time,
    )