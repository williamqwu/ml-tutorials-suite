import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
from PIL import Image

def __ml1_load_data(case: int, mode: str = "train"):
    datapath = f".tmp/ml_1/data_case{case}.pt"
    data = torch.load(datapath)
    x_train, y_train = data["x_train"], data["y_train"]
    assert x_train.shape[1] == 2
    x_np = x_train.numpy()
    y_np = y_train.numpy()
    return x_np, y_np

def __ml1_load_w(case: int):
    datapath = f".tmp/ml_1/stats_case{case}.npz"
    w = np.load(datapath)["w_hist"]
    return w

def ml1_show_dataset_2d(case: int):
    x_np, y_np = __ml1_load_data(case, mode="train")
    pos = y_np > 0
    neg = y_np < 0

    plt.figure(figsize=(6,6), dpi=200)
    plt.scatter(x_np[pos, 0], x_np[pos, 1], c='blue', label='Positive', alpha=0.6)
    plt.scatter(x_np[neg, 0], x_np[neg, 1], c='red', label='Negative', alpha=0.6)

    plt.title("Visualization of Generated Dataset")
    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.tight_layout()
    plt.show()


def ml1_gen_w_seq(case: int, lastepoch: int = -1):
    x_np, y_np = __ml1_load_data(case, mode="train")
    w_hist = __ml1_load_w(case)
    
    if lastepoch==-1:
        epochs = w_hist.shape[0]
    else:
        epochs = min(lastepoch, w_hist.shape[0])

    x_min, x_max = x_np[:, 0].min() - 1, x_np[:, 0].max() + 1
    y_min, y_max = x_np[:, 1].min() - 1, x_np[:, 1].max() + 1
    print(x_min,x_max,y_min,y_max)

    for epoch in range(epochs):
        plt.figure(figsize=(6, 6),dpi=200)
        x_seen = x_np[:epoch+1]
        y_seen = y_np[:epoch+1]
        pos = y_seen > 0
        neg = y_seen < 0
        plt.scatter(x_seen[pos, 0], x_seen[pos, 1], color="blue", label="Positive", alpha=0.6)
        plt.scatter(x_seen[neg, 0], x_seen[neg, 1], color="red", label="Negative", alpha=0.6)

        w = w_hist[epoch]
        x_vals = np.array([x_min, x_max])
        if abs(w[1]) < 1e-8:
            plt.axvline(x=0, color='#00B050', linestyle='--', label="Decision boundary")
        else:
            slope = -w[0] / w[1]
            intercept = 0
            y_vals = slope * x_vals + intercept
            plt.plot(x_vals, y_vals, color='#00B050', linestyle='--', label="Decision boundary")
        plt.title(f"Epoch {epoch+1}")
        plt.xlabel("$x_1$")
        plt.ylabel("$x_2$")
        plt.legend()
        plt.grid(True)
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.tight_layout()

        outputpath = f".tmp/ml_1/fig_case{case}"
        os.makedirs(outputpath, exist_ok=True)
        plt.savefig(os.path.join(outputpath, f"w_e{epoch+1:03d}.png"))
        plt.close()

def ml1_animate(case: int, fps: int = 2):
    figpath = f".tmp/ml_1/fig_case{case}"
    frame_files = sorted([
        os.path.join(figpath, f) for f in os.listdir(figpath)
        if f.endswith(".png")
    ])

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