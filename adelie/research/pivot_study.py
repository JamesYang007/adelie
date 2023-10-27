import adelie as ad
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
from adelie.logger import logger
from IPython.display import HTML


root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
tex_path = os.path.join(root_path, "docs/tex/pivot_rule")
fig_path = os.path.join(tex_path, "figures")


def active_sets(state):
    p = state.X.cols()
    betas = state.betas
    feature_to_group = np.empty(p, dtype=int)
    for i, (g, gs) in enumerate(zip(state.groups, state.group_sizes)):
        feature_to_group[g:g+gs] = i
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

        if "safe" in method:
            predict_set = predict_set.intersection(safe_sets[i])

        if "lazy" in method:
            previous_predict_set = predict_set if i == 0 else predict_sets[i-1]
            if active_sets[i+1].issubset(previous_predict_set):
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
        safe_set,
        delta,
    ):
        if "safe" in method:
            slack_set = safe_set
        else:
            slack_set = set(np.arange(G))
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
                    predict_set = _fixed_update(
                        previous_predict_set,
                        safe_sets[i],
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
                predict_set = _fixed_update(
                    predict_set,
                    safe_sets[i],
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
