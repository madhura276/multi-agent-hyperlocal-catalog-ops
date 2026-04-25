from pathlib import Path

import matplotlib.pyplot as plt


OUTPUT_DIR = Path("outputs/plots")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BASELINE_SCORE = 0.3323
TRAINED_SCORE = 0.3761

steps = [1, 2, 3, 4, 5, 6, 7, 8]
losses = [1.315472, 1.183308, 1.133816, 0.955122, 0.924255, 0.922068, 0.837844, 0.959402]
token_accuracy = [
    0.7170087993,
    0.7370479107,
    0.7361316681,
    0.7634408772,
    0.7693059742,
    0.7748579681,
    0.7876113951,
    0.7704280019,
]


def save_loss_plot() -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(steps, losses, marker="o", linewidth=2)
    plt.title("Training Loss During Post-Training")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "training_loss.png", dpi=200)
    plt.close()


def save_accuracy_plot() -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(steps, token_accuracy, marker="o", linewidth=2, color="green")
    plt.title("Mean Token Accuracy During Post-Training")
    plt.xlabel("Step")
    plt.ylabel("Mean Token Accuracy")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "token_accuracy.png", dpi=200)
    plt.close()


def save_score_comparison_plot() -> None:
    improvement = TRAINED_SCORE - BASELINE_SCORE
    labels = ["Baseline", "Trained"]
    values = [BASELINE_SCORE, TRAINED_SCORE]
    colors = ["#6c5ce7", "#00b894"]

    plt.figure(figsize=(6, 5))
    bars = plt.bar(labels, values, color=colors)
    plt.title("Before vs After Training Score")
    plt.ylabel("Average Score")
    plt.ylim(0, max(values) + 0.1)

    for bar, val in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            val + 0.01,
            f"{val:.4f}",
            ha="center",
        )

    plt.text(
        0.5,
        max(values) + 0.05,
        f"Improvement: +{improvement:.4f}",
        ha="center",
    )

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "score_comparison.png", dpi=200)
    plt.close()


def main() -> None:
    save_loss_plot()
    save_accuracy_plot()
    save_score_comparison_plot()
    print(f"Saved plots to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
