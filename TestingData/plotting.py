import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ORDER = ["SI1", "SI5", "SI1000", "contact_change"]


def load_data(task_name, base_dir="iLQR_AD"):
    task_path = os.path.join(base_dir, task_name)
    if not os.path.isdir(task_path):
        raise FileNotFoundError(f"Task '{task_name}' not found.")

    keypoint_methods = [
        d for d in os.listdir(task_path)
        if os.path.isdir(os.path.join(task_path, d))
    ]

    data = {}
    for method in keypoint_methods:
        csv_path = os.path.join(task_path, method, "summary.csv")
        if not os.path.isfile(csv_path):
            print(f"Warning: Missing summary.csv for {method}, skipping.")
            continue

        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"Error reading {csv_path}: {e}")
            continue

        df.columns = [c.strip() for c in df.columns]
        data[method] = df

    return data


def compute_stats(data):
    """Compute mean and 95% CI for relevant metrics."""
    results = {}

    for method, df in data.items():
        results[method] = {}

        for col in [
            "Task number Cost reduction",
            "Optimisation time (ms)",
            "Number iterations"
        ]:
            values = df[col].values
            mean = np.mean(values)
            ci = 1.96 * np.std(values, ddof=1) / np.sqrt(len(values))

            results[method][col] = (mean, ci)

    return results


def plot_bars(results):
    methods = list(results.keys())

    # Extract means & CIs
    def extract(col):
        means = [results[m][col][0] for m in methods]
        cis = [results[m][col][1] for m in methods]
        return means, cis

    cost_mean, cost_ci = extract("Task number Cost reduction")
    time_mean, time_ci = extract("Optimisation time (ms)")
    iter_mean, iter_ci = extract("Number iterations")

    fig, axs = plt.subplots(3, 1, figsize=(10, 12))

    # Cost reduction
    axs[0].bar(methods, cost_mean, yerr=cost_ci, capsize=5)
    axs[0].set_title("Cost Reduction (mean ± 95% CI)")
    axs[0].grid(True, axis='y')

    # Optimisation time
    axs[1].bar(methods, time_mean, yerr=time_ci, capsize=5)
    axs[1].set_title("Optimisation Time (ms) (mean ± 95% CI)")
    axs[1].grid(True, axis='y')

    # Number iterations
    axs[2].bar(methods, iter_mean, yerr=iter_ci, capsize=5)
    axs[2].set_title("Number Iterations (mean ± 95% CI)")
    axs[2].grid(True, axis='y')

    plt.tight_layout()
    plt.show()


def print_latex_table(results):
    print("\n================ LATEX TABLE =================")

    print("\nMethod & Cost Reduction (mean ± CI) & Time (ms) (mean ± CI) \\\\")
    
    for method in ORDER:
        if method not in results:
            continue

        print("{method} & ")
    print("\\\\")

    for method in ORDER:
        if method not in results:
            continue

        cr_mean, cr_ci = results[method]["Task number Cost reduction"]
        t_mean, t_ci = results[method]["Optimisation time (ms)"]
        
        print(f"{t_mean:.2f} & {cr_mean:.2f} $\pm$ {cr_ci:.2f} &", end=" ")
        
    print("\\\\")

    print("==============================================\n")


def main(task_name, base_dir):
    # Load raw data
    data = load_data(task_name, base_dir)
    if not data:
        print("No valid data found.")
        return

    # Compute summary statistics
    results = compute_stats(data)

    # Plot bar charts
    plot_bars(results)

    # Print LaTeX-friendly table
    print_latex_table(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("task", type=str, help="Task name")
    parser.add_argument("--base", type=str, default="iLQR_AD")
    args = parser.parse_args()

    main(args.task, args.base)
