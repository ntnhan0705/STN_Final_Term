import os, glob, csv
from pathlib import Path

def _xywhn_to_xyxy(xywhn):
    # xywh normalized -> xyxy normalized
    x, y, w, h = xywhn
    x1 = x - w / 2.0
    y1 = y - h / 2.0
    x2 = x + w / 2.0
    y2 = y + h / 2.0
    return [x1, y1, x2, y2]

def _iou(a, b):
    # a,b: [x1,y1,x2,y2] normalized
    xa1, ya1, xa2, ya2 = a
    xb1, yb1, xb2, yb2 = b
    inter_x1, inter_y1 = max(xa1, xb1), max(ya1, yb1)
    inter_x2, inter_y2 = min(xa2, xb2), min(ya2, yb2)
    iw, ih = max(0.0, inter_x2 - inter_x1), max(0.0, inter_y2 - inter_y1)
    inter = iw * ih
    if inter == 0: return 0.0
    area_a = max(0.0, xa2 - xa1) * max(0.0, ya2 - ya1)
    area_b = max(0.0, xb2 - xb1) * max(0.0, yb2 - yb1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0

def read_label_file(p):
    items = []
    if not os.path.exists(p): return items
    with open(p, "r", encoding="utf-8") as f:
        for ln in f:
            toks = ln.strip().split()
            if len(toks) < 5:
                continue
            # pred.txt có 6 trường: cls x y w h conf -> bỏ conf
            cls = int(float(toks[0]))
            x,y,w,h = map(float, toks[1:5])
            items.append((cls, _xywhn_to_xyxy([x,y,w,h])))
    return items

def main():
    # chỉnh 2 đường dẫn dưới cho đúng run hiện tại
    gt_labels = Path(r"C:/OneDrive/Study/AI/STN_Final_Term/datasets/valid/labels")
    pred_labels = Path(r"runs_stn/yolo11m_stn_run122/val/labels")

    stems = set([Path(p).stem for p in glob.glob(str(gt_labels/"*.txt"))])
    out_csv = pred_labels.parent / "val_probe_epoch.csv"

    rows = [["stem","n_gt","n_pred","mean_best_iou","gt_>0.1","gt_>0.3","gt_>0.5"]]
    for s in sorted(stems):
        gts = read_label_file(gt_labels / f"{s}.txt")
        prs = read_label_file(pred_labels / f"{s}.txt")
        if len(gts) == 0:
            rows.append([s,0,len(prs),0.0,0.0,0.0,0.0])
            continue
        bests = []
        for _, g in gts:
            best = 0.0
            for _, p in prs:
                best = max(best, _iou(g,p))
            bests.append(best)
        mean_iou = sum(bests)/len(bests) if bests else 0.0
        gt10 = sum(i>0.1 for i in bests)/max(1,len(bests))
        gt30 = sum(i>0.3 for i in bests)/max(1,len(bests))
        gt50 = sum(i>0.5 for i in bests)/max(1,len(bests))
        rows.append([s, len(gts), len(prs), round(mean_iou,4), round(gt10,3), round(gt30,3), round(gt50,3)])

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)
    print(f"[Probe] wrote {out_csv}")

if __name__ == "__main__":
    main()
