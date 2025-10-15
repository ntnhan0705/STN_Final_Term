# Check_data_flow.py
import sys
from pathlib import Path
import torch

# 1) Đặt đường dẫn đến project root và thư mục ultralytics vào sys.path
project_root = Path(r"/")
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "ultralytics"))

# 2) Import v8DetectionLoss để khởi tạo instance "empty" cho warp_bbox
from ultralytics.utils.loss import v8DetectionLoss

# Mở rộng print để không bị cắt tensor lớn
torch.set_printoptions(threshold=int(1e9), precision=6, linewidth=200)

def tensor_stats(name: str, t: torch.Tensor, print_full: bool = False):
    print(f"--- {name} ---")
    print(f" shape: {tuple(t.shape)}")
    print(f" dtype: {t.dtype}")
    nan_cnt  = int(torch.isnan(t).sum())
    zero_cnt = int((t == 0).sum())
    print(f" nan count: {nan_cnt}")
    print(f" zero count: {zero_cnt} / {t.numel()}")
    if nan_cnt == 0:
        print(f" min: {float(torch.min(t))}, max: {float(torch.max(t))}")
    else:
        print(" min,max: skipped because NaNs present")
    # In full tensor nếu được bật
    if print_full:
        print(f"\n>> FULL {name} matrix:")
        print(t)
    else:
        # In các dòng không phải all-zero (dễ kiểm tra)
        if t.ndim >= 2:
            nonzero_rows = (t.abs().sum(dim=-1) != 0)
            for batch_idx in range(t.shape[0]):
                rows = t[batch_idx]
                valid = rows[nonzero_rows[batch_idx]]
                print(f" batch {batch_idx} valid rows:")
                print(valid if valid.numel() else "[no valid rows]")
        else:
            print(" sample:", t)

    print()

def main():
    debug_dir = project_root / "scripts" / "debug_stn"

    # 3) Load debug data
    gt_pre  = torch.load(debug_dir / "debug_gt_prewarp.pt")   # [B, M, 4]
    labels  = torch.load(debug_dir / "debug_labels.pt")       # [B, M, 1]
    theta   = torch.load(debug_dir / "debug_theta.pt")        # [B, 2, 3]
    gt_post = torch.load(debug_dir / "debug_gt_postwarp.pt")  # [B, M, 4]
    with open(debug_dir / "debug_im_files.txt", "r") as f:
        im_files = [line.strip() for line in f]

    # In đường dẫn hình ảnh
    print("=== DEBUG IM FILES ===")
    for i,p in enumerate(im_files):
        print(f" [{i}] {p}")
    print()

    # 4) In stats cho từng tensor
    tensor_stats("GT boxes BEFORE warp", gt_pre,  print_full=False)
    tensor_stats("GT labels",             labels,  print_full=False)
    tensor_stats("Theta",                 theta,   print_full=True)   # **in full ma trận 2×3**
    tensor_stats("GT boxes AFTER warp",   gt_post, print_full=False)

    # 5) Thử gọi warp_bbox trực tiếp để kiểm tra hàm
    loss = v8DetectionLoss.__new__(v8DetectionLoss)
    loss.device = torch.device("cpu")
    class H: box=cls=dfl=1.0
    loss.hyp = H()

    imgsz = torch.tensor([640., 640.], dtype=torch.float32)
    warped_again = v8DetectionLoss.warp_bbox(loss, gt_pre, theta, imgsz)
    tensor_stats("Warped AGAIN (from GT_PRE)", warped_again, print_full=False)

    # 6) Lưu kết quả test lần cuối nếu muốn
    torch.save(warped_again, debug_dir / "debug_warped_again.pt")
    print(f"Saved re-warped to: {debug_dir/'debug_warped_again.pt'}")

if __name__ == "__main__":
    main()
