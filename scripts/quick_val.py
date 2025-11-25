"""Quick validation for trained checkpoint with default Ultralytics settings."""
from pathlib import Path
from ultralytics import YOLO


def main():
    base_dir = Path(__file__).resolve().parent.parent
    weights_path = base_dir / "runs" / "yolo11m_stn_run001" / "weights" / "epoch2.pt"
    data_path = base_dir / "dataset.yaml"

    if not weights_path.exists():
        raise FileNotFoundError(f"Khong tim thay file weights: {weights_path.resolve()}")
    if not data_path.exists():
        raise FileNotFoundError(f"Khong tim thay file data: {data_path.resolve()}")

    model = YOLO(str(weights_path))
    metrics = model.val(data=str(data_path))

    results_dict = getattr(metrics, "results_dict", None)
    if callable(results_dict):
        print(results_dict())
    elif results_dict is not None:
        print(results_dict)
    else:
        print(metrics)


if __name__ == "__main__":
    main()
