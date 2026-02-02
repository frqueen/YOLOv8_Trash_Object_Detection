from ultralytics import YOLO

def main():

    # Load pretrained YOLOv8 nano model
    model = YOLO("yolov8n.pt")

    # Train model
    model.train(
        data="/kaggle/input/trash-yolo/data.yaml",
        epochs=30,
        imgsz=640,
        batch=32,
        optimizer="SGD",
        lr0=0.0001,
        momentum=0.9,
        weight_decay=0.0005,
        warmup_epochs=5,
        warmup_momentum=0.8,
        warmup_bias_lr=0.2,
        box=0.05,
        cls=0.08,
        dfl=0.1,
        close_mosaic=7,
        iou=0.6,
        plots=True
    )

if __name__ == "__main__":
    main()
import shutil
shutil.make_archive("runs_backup", "zip", "runs")
