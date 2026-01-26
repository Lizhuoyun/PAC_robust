import json
import os
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


def plot_curve(budgets: List[float], scores: List[float], out_path: str, title: str, ylabel: str) -> None:
    plt.figure()
    plt.plot(budgets, scores, marker="o")
    plt.xlabel("Perturbation budget")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_correlation(x: List[float], y: List[float], out_path: str, xlabel: str, ylabel: str) -> None:
    plt.figure()
    plt.scatter(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title("Correlation")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_heatmap(matrix: List[List[float]], out_path: str, title: str) -> None:
    plt.figure()
    mat = np.array(matrix)
    plt.imshow(mat, cmap="viridis")
    plt.colorbar()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def load_matrix(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_heatmap(matrix_path: str, out_path: str, title: str) -> None:
    data = load_matrix(matrix_path)
    plot_heatmap(data["matrix_gamma"], out_path, title)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--title", default="Transition Matrix")
    args = parser.parse_args()
    save_heatmap(args.matrix, args.out, args.title)
