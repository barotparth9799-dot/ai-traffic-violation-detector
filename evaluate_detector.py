import csv
from pathlib import Path

from detector import TrafficViolationDetector


PROJECT_DIR = Path(__file__).resolve().parent

IMAGE_DIR = PROJECT_DIR / "evaluation_dataset" / "test" / "images"
LABEL_DIR = PROJECT_DIR / "evaluation_dataset" / "test" / "labels"
RESULT_DIR = PROJECT_DIR / "evaluation_results"

RESULT_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = RESULT_DIR / "evaluation_results.csv"
SUMMARY_PATH = RESULT_DIR / "evaluation_summary.txt"


def read_ground_truth(label_path):
    """
    Read YOLO labels.

    Dataset classes:
        0 = Helmet
        1 = NOHelmet
        2 = Triple Riding
    """

    no_helmet = False
    triple_riding = False

    if not label_path.exists():
        return no_helmet, triple_riding

    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()

            if not parts:
                continue

            try:
                class_id = int(parts[0])
            except ValueError:
                continue

            if class_id == 1:
                no_helmet = True

            elif class_id == 2:
                triple_riding = True

    return no_helmet, triple_riding


def get_predictions(violations):
    """
    Convert detector output into image-level predictions.
    """

    no_helmet = any(
        item.get("violation_type") == "Riding without Helmet"
        for item in violations
    )

    triple_riding = any(
        item.get("violation_type") == "Triple Riding"
        for item in violations
    )

    return no_helmet, triple_riding


def calculate_metrics(ground_truth, predictions):
    tp = tn = fp = fn = 0

    for actual, predicted in zip(ground_truth, predictions):

        if actual and predicted:
            tp += 1

        elif not actual and not predicted:
            tn += 1

        elif not actual and predicted:
            fp += 1

        elif actual and not predicted:
            fn += 1

    total = tp + tn + fp + fn

    accuracy = (
        (tp + tn) / total
        if total else 0.0
    )

    precision = (
        tp / (tp + fp)
        if (tp + fp) else 0.0
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn) else 0.0
    )

    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) else 0.0
    )

    return {
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
    }


def print_metrics(title, metrics):
    print()
    print("=" * 60)
    print(title)
    print("=" * 60)

    print("TP:", metrics["TP"])
    print("TN:", metrics["TN"])
    print("FP:", metrics["FP"])
    print("FN:", metrics["FN"])

    print(
        "Accuracy : {:.2f}%".format(
            metrics["Accuracy"] * 100
        )
    )

    print(
        "Precision: {:.2f}%".format(
            metrics["Precision"] * 100
        )
    )

    print(
        "Recall   : {:.2f}%".format(
            metrics["Recall"] * 100
        )
    )

    print(
        "F1 Score : {:.2f}%".format(
            metrics["F1"] * 100
        )
    )


def main():

    if not IMAGE_DIR.exists():
        raise FileNotFoundError(
            f"Image directory not found: {IMAGE_DIR}"
        )

    if not LABEL_DIR.exists():
        raise FileNotFoundError(
            f"Label directory not found: {LABEL_DIR}"
        )

    image_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".webp",
    }

    image_files = sorted(
        [
            p for p in IMAGE_DIR.iterdir()
            if p.is_file()
            and p.suffix.lower() in image_extensions
        ]
    )

    print("=" * 60)
    print("TRAFFIC VIOLATION DETECTOR - INDEPENDENT EVALUATION")
    print("=" * 60)

    print("Evaluation images:", len(image_files))
    print("Image directory:", IMAGE_DIR)
    print("Label directory:", LABEL_DIR)
    print()

    if len(image_files) == 0:
        raise RuntimeError("No evaluation images found.")

    detector = TrafficViolationDetector()

    results = []

    no_helmet_actual = []
    no_helmet_predicted = []

    triple_actual = []
    triple_predicted = []

    for index, image_path in enumerate(image_files, start=1):

        label_path = LABEL_DIR / (
            image_path.stem + ".txt"
        )

        actual_no_helmet, actual_triple = (
            read_ground_truth(label_path)
        )

        try:
            annotated, violations = (
                detector.detect_violations(
                    str(image_path)
                )
            )

            predicted_no_helmet, predicted_triple = (
                get_predictions(violations)
            )

            error = ""

        except Exception as exc:

            predicted_no_helmet = False
            predicted_triple = False

            error = str(exc)

        no_helmet_actual.append(actual_no_helmet)
        no_helmet_predicted.append(predicted_no_helmet)

        triple_actual.append(actual_triple)
        triple_predicted.append(predicted_triple)

        results.append(
            {
                "filename": image_path.name,
                "ground_truth_no_helmet":
                    int(actual_no_helmet),
                "predicted_no_helmet":
                    int(predicted_no_helmet),
                "ground_truth_triple_riding":
                    int(actual_triple),
                "predicted_triple_riding":
                    int(predicted_triple),
                "total_violations":
                    len(violations)
                    if error == ""
                    else 0,
                "error": error,
            }
        )

        print(
            "[{}/{}] {}".format(
                index,
                len(image_files),
                image_path.name
            )
        )

        print(
            "    GT: NoHelmet={}, Triple={}".format(
                int(actual_no_helmet),
                int(actual_triple)
            )
        )

        print(
            "    Pred: NoHelmet={}, Triple={}".format(
                int(predicted_no_helmet),
                int(predicted_triple)
            )
        )

        if error:
            print("    ERROR:", error)

    # ---------------------------------------------------------
    # Calculate metrics
    # ---------------------------------------------------------

    no_helmet_metrics = calculate_metrics(
        no_helmet_actual,
        no_helmet_predicted
    )

    triple_metrics = calculate_metrics(
        triple_actual,
        triple_predicted
    )

    # ---------------------------------------------------------
    # Save detailed CSV
    # ---------------------------------------------------------

    with open(
        CSV_PATH,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "filename",
                "ground_truth_no_helmet",
                "predicted_no_helmet",
                "ground_truth_triple_riding",
                "predicted_triple_riding",
                "total_violations",
                "error",
            ]
        )

        writer.writeheader()
        writer.writerows(results)

    # ---------------------------------------------------------
    # Save summary
    # ---------------------------------------------------------

    summary_lines = []

    summary_lines.append(
        "TRAFFIC VIOLATION DETECTOR - EVALUATION SUMMARY"
    )

    summary_lines.append("=" * 60)

    summary_lines.append(
        f"Total test images: {len(image_files)}"
    )

    summary_lines.append("")

    for title, metrics in [
        ("NO HELMET", no_helmet_metrics),
        ("TRIPLE RIDING", triple_metrics),
    ]:

        summary_lines.append(title)
        summary_lines.append("-" * 60)

        summary_lines.append(
            f"TP: {metrics['TP']}"
        )

        summary_lines.append(
            f"TN: {metrics['TN']}"
        )

        summary_lines.append(
            f"FP: {metrics['FP']}"
        )

        summary_lines.append(
            f"FN: {metrics['FN']}"
        )

        summary_lines.append(
            "Accuracy: {:.2f}%".format(
                metrics["Accuracy"] * 100
            )
        )

        summary_lines.append(
            "Precision: {:.2f}%".format(
                metrics["Precision"] * 100
            )
        )

        summary_lines.append(
            "Recall: {:.2f}%".format(
                metrics["Recall"] * 100
            )
        )

        summary_lines.append(
            "F1 Score: {:.2f}%".format(
                metrics["F1"] * 100
            )
        )

        summary_lines.append("")

    with open(
        SUMMARY_PATH,
        "w",
        encoding="utf-8"
    ) as f:

        f.write("\n".join(summary_lines))

    # ---------------------------------------------------------
    # Console summary
    # ---------------------------------------------------------

    print()
    print()
    print("#" * 60)
    print("FINAL EVALUATION RESULTS")
    print("#" * 60)

    print_metrics(
        "NO HELMET DETECTION",
        no_helmet_metrics
    )

    print_metrics(
        "TRIPLE RIDING DETECTION",
        triple_metrics
    )

    print()
    print("=" * 60)
    print("FILES SAVED")
    print("=" * 60)

    print("Detailed results:")
    print(CSV_PATH)

    print()
    print("Summary:")
    print(SUMMARY_PATH)

    print()
    print("Evaluation completed.")


if __name__ == "__main__":
    main()