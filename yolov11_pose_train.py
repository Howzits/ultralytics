from ultralytics.models.yolo import YOLO

model = YOLO("./runs/pose/yolo11n-pose-Baseline/weights/best.pt")

model.train(
    data="/99_TemporaryData/haochen75/edge_detect/book-keypoints.yaml",
    task="pose",
    name="yolo11n-pose-Baseline_augmented",
    epochs=100,
    batch=64,
    workers=16,
    device=["0"],
    imgsz=320,
    optimizer="Adam",
    lr0=1e-4,
    lrf=0.01,
    mosaic=0.0,
    box=2.0,
    pose=15.0,
    kobj=1.0,
    dfl=0.1,
    cls=0.1,
    scale=0.25,
    degrees=20.0,
    translate=0.1,
    cos_lr=True,
    conf=0.9,
    iou=0.8,
)
