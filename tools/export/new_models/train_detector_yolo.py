"""Fine-tune YOLO11n as an angle-robust person detector for ZoneGuard-AI."""

from ultralytics import YOLO


def main():
    model = YOLO("yolo11n.pt")  # pretrained; already strong on 'person'

    model.train(
        data="tools/export/new_models/coco-person.yaml",
        classes=[0],  # person only
        imgsz=384,  # match the pipeline / manifest input_size
        epochs=80,
        batch=64,
        degrees=35,  # rotation aug -> robust to tilted/fallen people
        flipud=0.5,
        fliplr=0.5,
        scale=0.5,
        perspective=0.0005,
        mosaic=1.0,
        close_mosaic=10,
        name="zoneguard_ai",
    )


if __name__ == "__main__":
    main()
