from ultralytics.models.yolo import YOLO

# model = YOLO(
#     "/CodeRepo/Code/haochen75/workspace/ultralytics/runs/pose/yolo11n_pose_baseline_full_augmented6/weights/best.pt",
# )

model = YOLO(
    "yolo11n-pose.yaml",
)

model.train(
    data="/99_TemporaryData/haochen75/edge_detect/old/book-keypoints.yaml",
    task="pose",
    name="yolo11n_pose_new_block",
    epochs=10,
    batch=64,
    workers=64,
    device=[2],
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
    fliplr=0.0,
    flipud=0.0,
    cos_lr=True,
    conf=0.9,
    iou=0.8,
    # patience=0,
    amp=False,
    auto_augment=None,
    # val = False,
)

# model = YOLO("./runs/pose/yolo11n-pose-Baseline/weights/best.pt")
# metrics = model.val(
#     data="/99_TemporaryData/haochen75/edge_detect/new/book-keypoints.yaml",
#     task="test",
#     conf=0.9,
#     iou=0.5,
#     device="0",
#     imgsz=320,
#     rect=True,
#     batch=64,
#     save_json=True,
#     save = True,
#     plots = True,
#     save_crop = True,
#     # save_txt = True,
# )
