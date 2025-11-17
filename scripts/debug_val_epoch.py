from ultralytics import YOLO

# Đổi path này sang weights epoch 2 của bạn
model_path = r"C:/OneDrive/Study/AI/STN_Final_Term/runs/yolo11m_stn_run00145/weights/epoch2.pt"
data_yaml = r"C:/OneDrive/Study/AI/STN_Final_Term/dataset.yaml"

model = YOLO(model_path)
metrics = model.val(
    data=data_yaml,
    imgsz=640,
    split="val",
    conf=0.01,
    iou=0.10,
    max_det=300,
)
print(metrics)
