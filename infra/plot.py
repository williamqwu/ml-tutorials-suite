import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import matplotlib.animation as animation
from IPython.display import HTML
from PIL import Image
from sklearn.linear_model import SGDClassifier
import sympy

CFG_ML1 = {
    "c_pos": "blue",
    "c_neg": "red",
    "c_lds": "#00B050",
    "path_prefix": ".tmp/ml_1",
}

CFG_ML2 = {
    "c_pos": "blue",
    "c_neg": "red",
    "c_lds": "#00B050",
    "c_loss": "orange",
    "c_acc": "#3399FF",
    "c_gd": "#3399FF",
    "c_sgd": "#00B050",
    "path_prefix": ".tmp/ml_2",
}


def __ml1_load_data(case: int, mode: str = "train"):
    datapath = os.path.join(CFG_ML1["path_prefix"], f"data_case{case}.pt")
    data = torch.load(datapath)
    x_train, y_train = data["x_train"], data["y_train"]
    assert x_train.shape[1] == 2
    x_np = x_train.numpy()
    y_np = y_train.numpy()
    return x_np, y_np


def __ml1_load_w(case: int):
    datapath = os.path.join(CFG_ML1["path_prefix"], f"stats_case{case}.npz")
    w = np.load(datapath)["w_hist"]
    return w


def ml1_calc_margin(case: int):
    x_np, y_np = __ml1_load_data(case, mode="train")
    R = np.max(np.linalg.norm(x_np, axis=1))
    # Warning: for demo purpose only -
    #   this part doesn't strictly follow the slides definition
    clf = SGDClassifier(loss="hinge", alpha=1e-6, fit_intercept=False)
    clf.fit(x_np, y_np)
    w = clf.coef_[0]
    w_norm = np.linalg.norm(w)
    gamma = np.min(y_np * (x_np @ w)) / w_norm
    mistake_bound = (R / gamma) ** 2

    print(f"R = {R:.4f}")
    print(f"gamma = {gamma:.4f}")
    print(f"Mistake <= {mistake_bound:.2f}")


def ml1_show_dataset_2d(case: int, margin: bool = False):
    x_np, y_np = __ml1_load_data(case, mode="train")
    pos = y_np > 0
    neg = y_np < 0
    x_min, x_max = x_np[:, 0].min() - 1, x_np[:, 0].max() + 1
    y_min, y_max = x_np[:, 1].min() - 1, x_np[:, 1].max() + 1

    plt.figure(figsize=(6, 6), dpi=200)
    plt.scatter(
        x_np[pos, 0], x_np[pos, 1], c=CFG_ML1["c_pos"], label="Positive", alpha=0.6
    )
    plt.scatter(
        x_np[neg, 0], x_np[neg, 1], c=CFG_ML1["c_neg"], label="Negative", alpha=0.6
    )

    if margin:
        R = np.max(np.linalg.norm(x_np, axis=1))
        y_np_margin = np.where(y_np > 0, 1, -1)
        clf = SGDClassifier(loss="hinge", alpha=1e-6, fit_intercept=False)
        clf.fit(x_np, y_np_margin)
        w = clf.coef_[0]
        w_norm = np.linalg.norm(w)
        gamma = np.min(y_np_margin * (x_np @ w)) / w_norm
        slope = -w[0] / (w[1] + 1e-6)
        x_vals = np.linspace(x_np[:, 0].min() - 1, x_np[:, 0].max() + 1, 200)
        y_center = slope * x_vals

        norm_vec = np.array([-w[1], w[0]]) / w_norm
        offset = gamma * norm_vec

        plt.plot(
            x_vals,
            y_center,
            color=CFG_ML1["c_lds"],
            linestyle="--",
            label="Decision Boundary",
        )
        plt.plot(x_vals, y_center + offset[1], color=CFG_ML1["c_lds"], linestyle=":")
        plt.plot(
            x_vals,
            y_center - offset[1],
            color=CFG_ML1["c_lds"],
            linestyle=":",
            alpha=0.6,
        )
        plt.fill_between(
            x_vals,
            y_center - offset[1],
            y_center + offset[1],
            color=CFG_ML1["c_lds"],
            alpha=0.1,
            label="Margin Band (±$\\gamma$)",
        )

        ball = Circle(
            (0, 0),
            R,
            color="gray",
            linestyle="--",
            fill=False,
            alpha=0.2,
            label="Radius R",
        )
        plt.gca().add_patch(ball)

        plt.title(
            f"Dataset with Margin & Radius\n $\\gamma$ = {gamma:.2f}, R = {R:.2f}, Err ≤ {(R / gamma) ** 2:.1f}"
        )
    else:
        plt.title(f"Visualization of Generated Dataset (Case: {case})")

    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")
    plt.legend()
    plt.grid(True)
    # plt.axis('equal')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.tight_layout()
    plt.show()


def ml1_gen_w_seq(case: int, lastepoch: int = -1):
    x_np, y_np = __ml1_load_data(case, mode="train")
    w_hist = __ml1_load_w(case)

    if lastepoch == -1:
        epochs = w_hist.shape[0]
    else:
        epochs = min(lastepoch, w_hist.shape[0])

    x_min, x_max = x_np[:, 0].min() - 1, x_np[:, 0].max() + 1
    y_min, y_max = x_np[:, 1].min() - 1, x_np[:, 1].max() + 1
    # print(x_min,x_max,y_min,y_max)

    for epoch in range(epochs):
        plt.figure(figsize=(6, 6), dpi=200)
        x_seen = x_np[: epoch + 1]
        y_seen = y_np[: epoch + 1]
        pos = y_seen > 0
        neg = y_seen < 0
        plt.scatter(
            x_seen[pos, 0],
            x_seen[pos, 1],
            color=CFG_ML1["c_pos"],
            label="Positive",
            alpha=0.6,
        )
        plt.scatter(
            x_seen[neg, 0],
            x_seen[neg, 1],
            color=CFG_ML1["c_neg"],
            label="Negative",
            alpha=0.6,
        )

        w = w_hist[epoch]
        x_vals = np.array([x_min, x_max])
        if abs(w[1]) < 1e-8:
            plt.axvline(
                x=0, color=CFG_ML1["c_lds"], linestyle="--", label="Decision boundary"
            )
        else:
            slope = -w[0] / w[1]
            intercept = 0
            y_vals = slope * x_vals + intercept
            plt.plot(
                x_vals,
                y_vals,
                color=CFG_ML1["c_lds"],
                linestyle="--",
                label="Decision boundary",
            )
        plt.title(f"Phase {epoch + 1}")
        plt.xlabel("$x_1$")
        plt.ylabel("$x_2$")
        plt.legend()
        plt.grid(True)
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.gca().set_aspect("equal", adjustable="box")
        plt.tight_layout()

        outputpath = os.path.join(CFG_ML1["path_prefix"], f"fig_case{case}")
        os.makedirs(outputpath, exist_ok=True)
        plt.savefig(os.path.join(outputpath, f"w_e{epoch + 1:03d}.png"))
        plt.close()


def ml1_animate(case: int, fps: int = 2):
    figpath = os.path.join(CFG_ML1["path_prefix"], f"fig_case{case}")
    frame_files = sorted(
        [os.path.join(figpath, f) for f in os.listdir(figpath) if f.endswith(".png")]
    )

    frames = [Image.open(f) for f in frame_files]
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.axis("off")

    img = plt.imshow(frames[0])

    def update(i):
        img.set_data(frames[i])
        return [img]

    ani = animation.FuncAnimation(
        fig, update, frames=len(frames), interval=1000 // fps, blit=True
    )
    plt.close(fig)  # prevent duplicate display
    return HTML(ani.to_jshtml())


def __ml2_taylor_order_k(func_str: str, k: int, w0: float):
    """
    Input:
        `func_str`: a math function str
        `k`: order of taylor expansion
        `w0`: the expansion is around W=w0
    Reference: github.com/alochaus/taylor-series
    """
    x = sympy.symbols("W")
    f = sympy.sympify(func_str)  # create a SymPy expression
    # compute 1st to kth derivatives of f w.r.t. x
    derivatives = [sympy.diff(f, x, i) for i in range(1, k + 1)]

    # init taylor polynomial w/ 0th term, i.e., l(w0)
    taylor_poly = f.subs(x, w0)
    # add remaining terms
    for i in range(1, k + 1):
        term = (derivatives[i - 1].subs(x, w0)) * ((x - w0) ** i) / sympy.factorial(i)
        taylor_poly += term
    # returns a 3-tuple:
    #   original function f(x)
    #   k-th order taylor polynomial function
    #   a string representation of k-th taylor polynomial
    return (
        sympy.lambdify(x, f, "numpy"),
        sympy.lambdify(x, taylor_poly, "numpy"),
        str(taylor_poly),
    )


def ml2_show_taylor_order_k(func_str, k, w0, w_range=(-3, 3), resolution=500):
    f_func, taylor_func, poly_str = __ml2_taylor_order_k(func_str, k, w0)
    w_start, w_end = w_range
    w_vals = np.linspace(w_start, w_end, resolution)
    f_vals = f_func(w_vals)
    taylor_vals = taylor_func(w_vals)
    if np.isscalar(taylor_vals):
        taylor_vals = np.full_like(w_vals, fill_value=taylor_vals)

    plt.figure(figsize=(10, 5), dpi=200)
    plt.plot(w_vals, f_vals, label="Original Function $l$", color="black")
    plt.plot(
        w_vals,
        taylor_vals,
        "--",
        label=f"{k}-order Taylor Approx. at $W_0$={w0}",
        color="red",
    )
    plt.axvline(w0, linestyle=":", color="gray", label=f"Expansion Point $W_0$={w0}")

    plt.title(f"{k}-Order Taylor Approximation of l(W) at $W_0$={w0}")
    plt.xlabel("W")
    plt.ylabel("l(W)")
    plt.legend()
    plt.grid(True)
    plt.show()

    print(f"Taylor Polynomial (order {k}):")
    print(poly_str)


def __ml2_load_data(case: int):
    datapath = os.path.join(CFG_ML2["path_prefix"], f"data_case{case}.pt")
    data = torch.load(datapath)
    x_train, y_train = data["x_train"], data["y_train"]
    assert x_train.shape[1] == 2
    x_np = x_train.numpy()
    y_np = y_train.numpy()
    return x_np, y_np


def ml2_show_dataset_2d(case: int, margin: bool = False):
    x_np, y_np = __ml2_load_data(case)
    y_flat = y_np.squeeze()
    pos = y_flat == 1.0
    neg = y_flat == 0.0
    x_min, x_max = x_np[:, 0].min() - 1, x_np[:, 0].max() + 1
    y_min, y_max = x_np[:, 1].min() - 1, x_np[:, 1].max() + 1

    plt.figure(figsize=(6, 6), dpi=200)
    plt.scatter(
        x_np[pos, 0], x_np[pos, 1], c=CFG_ML2["c_pos"], label="Positive", alpha=0.6
    )
    plt.scatter(
        x_np[neg, 0], x_np[neg, 1], c=CFG_ML2["c_neg"], label="Negative", alpha=0.6
    )

    plt.title(f"Visualization of Generated Dataset (Case: {case})")

    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")
    plt.legend()
    plt.grid(True)
    # plt.axis('equal')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.tight_layout()
    plt.show()


def ml2_gen_w_seq(cfg, lastepoch: int = -1):
    exp_postfix = f"case{cfg['case']}_{cfg['tr_mode']}_{cfg['lr']}"
    x_np, y_np = __ml2_load_data(cfg["case"])
    y_flat = y_np.squeeze()
    pos = y_flat == 1.0
    neg = y_flat == 0.0
    datapath = os.path.join(CFG_ML2["path_prefix"], f"stats_{exp_postfix}.npz")
    w_hist = np.load(datapath)["w_hist"]

    if lastepoch == -1:
        epochs = w_hist.shape[0]
    else:
        epochs = min(lastepoch, w_hist.shape[0])

    x_min, x_max = x_np[:, 0].min() - 1, x_np[:, 0].max() + 1
    y_min, y_max = x_np[:, 1].min() - 1, x_np[:, 1].max() + 1

    for epoch in range(epochs):
        plt.figure(figsize=(6, 6), dpi=200)
        plt.scatter(
            x_np[pos, 0], x_np[pos, 1], c=CFG_ML2["c_pos"], label="Positive", alpha=0.6
        )
        plt.scatter(
            x_np[neg, 0], x_np[neg, 1], c=CFG_ML2["c_neg"], label="Negative", alpha=0.6
        )

        w = w_hist[epoch]
        w_vec = w[:2]
        b = w[2].item()

        x_vals = np.array([x_min, x_max])

        if abs(w_vec[1]) < 1e-8:
            plt.axvline(
                x=-b / w_vec[0],
                color=CFG_ML2["c_lds"],
                linestyle="--",
                label="Decision boundary",
            )
        else:
            slope = -w_vec[0] / w_vec[1]
            intercept = -b / w_vec[1]
            y_vals = slope * x_vals + intercept
            plt.plot(
                x_vals,
                y_vals,
                color=CFG_ML2["c_lds"],
                linestyle="--",
                label="Decision boundary",
            )
        plt.title(f"Phase {epoch + 1}")
        plt.xlabel("$x_1$")
        plt.ylabel("$x_2$")
        plt.legend()
        plt.grid(True)
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.gca().set_aspect("equal", adjustable="box")
        plt.tight_layout()

        outputpath = os.path.join(CFG_ML2["path_prefix"], f"fig_{exp_postfix}")
        os.makedirs(outputpath, exist_ok=True)
        plt.savefig(os.path.join(outputpath, f"w_e{epoch + 1:03d}.png"))
        plt.close()


def ml2_animate(cfg, fps: int = 2):
    exp_postfix = f"case{cfg['case']}_{cfg['tr_mode']}_{cfg['lr']}"
    figpath = os.path.join(CFG_ML2["path_prefix"], f"fig_{exp_postfix}")
    frame_files = sorted(
        [os.path.join(figpath, f) for f in os.listdir(figpath) if f.endswith(".png")]
    )

    frames = [Image.open(f) for f in frame_files]
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.axis("off")

    img = plt.imshow(frames[0])

    def update(i):
        img.set_data(frames[i])
        return [img]

    ani = animation.FuncAnimation(
        fig, update, frames=len(frames), interval=1000 // fps, blit=True
    )
    plt.close(fig)  # prevent duplicate display
    return HTML(ani.to_jshtml())


def ml2_show_stats(cfg):
    datapath = os.path.join(
        CFG_ML2["path_prefix"],
        f"stats_case{cfg['case']}_{cfg['tr_mode']}_{cfg['lr']}.npz",
    )
    stats = np.load(datapath)["stats"]
    loss = stats[:, 0]
    acc = stats[:, 1]

    epochs = np.arange(1, len(loss) + 1)

    fig, ax1 = plt.subplots(figsize=(6, 4), dpi=200)

    ax1.plot(
        epochs,
        loss,
        label="Training Loss",
        marker="v",
        markersize=3,
        color=CFG_ML2["c_loss"],
    )
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(True, which="both")
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()
    ax2.plot(
        epochs,
        acc,
        label="Training Accuracy",
        linestyle="--",
        marker="^",
        markersize=3,
        color=CFG_ML2["c_acc"],
    )
    ax2.set_ylabel("Accuracy")
    ax2.set_yscale("log")
    ax2.legend(loc="lower right")

    plt.title(
        f"Training Stats (Case={cfg['case']}, Mode={cfg['tr_mode']}, LR={cfg['lr']})"
    )
    plt.tight_layout()
    plt.show()


def show_trajectory(cfg1, cfg2):
    def load_w_hist(cfg):
        exp_postfix = f"case{cfg['case']}_{cfg['tr_mode']}_{cfg['lr']}"
        datapath = os.path.join(CFG_ML2["path_prefix"], f"stats_{exp_postfix}.npz")
        return np.load(datapath)["w_hist"]

    def get_label(cfg):
        return f"{cfg['tr_mode'].upper()} (LR={cfg['lr']})"

    x_np, y_np = __ml2_load_data(cfg1["case"])
    y_np_bin = 2 * y_np.squeeze() - 1
    w_hist_1 = load_w_hist(cfg1)
    w_hist_2 = load_w_hist(cfg2)

    all_w = np.vstack([w_hist_1[:, :2], w_hist_2[:, :2]])
    w0_vals = np.linspace(all_w[:, 0].min() - 1, all_w[:, 0].max() + 1, 100)
    w1_vals = np.linspace(all_w[:, 1].min() - 1, all_w[:, 1].max() + 1, 100)
    W0, W1 = np.meshgrid(w0_vals, w1_vals)
    Z = np.zeros_like(W0)

    for i in range(W0.shape[0]):
        for j in range(W0.shape[1]):
            w_vec = np.array([W0[i, j], W1[i, j]])
            b = 0
            logits = x_np @ w_vec + b
            Z[i, j] = np.mean(np.log(1 + np.exp(-y_np_bin * logits)))

    plt.figure(figsize=(8, 6), dpi=200)
    contour = plt.contour(W0, W1, Z, levels=20, cmap="Reds")
    plt.clabel(contour, inline=1, fontsize=8)

    plt.plot(
        w_hist_1[:, 0],
        w_hist_1[:, 1],
        marker="o",
        linewidth=1.5,
        color=CFG_ML2["c_gd"],
        markersize=3,
        label=get_label(cfg1),
    )
    plt.scatter(
        w_hist_1[0, 0], w_hist_1[0, 1], color="black", marker="x", label="Start ($w_0$)"
    )
    plt.scatter(
        w_hist_1[-1, 0],
        w_hist_1[-1, 1],
        color=CFG_ML2["c_gd"],
        marker="*",
        label="End GD",
    )
    plt.plot(
        w_hist_2[:, 0],
        w_hist_2[:, 1],
        marker="^",
        linewidth=1.5,
        color=CFG_ML2["c_sgd"],
        markersize=3,
        label=get_label(cfg2),
    )
    plt.scatter(
        w_hist_2[-1, 0],
        w_hist_2[-1, 1],
        color=CFG_ML2["c_sgd"],
        marker="*",
        label="End SGD",
    )

    plt.xlabel(r"$w_0$")
    plt.ylabel(r"$w_1$")
    plt.title(f"Weight Trajectory Comparison (Case {cfg1['case']})")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
