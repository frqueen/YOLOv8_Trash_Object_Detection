from ultralytics import YOLO

def main():

    model = YOLO("weights/best.pt")   # path to trained model

    results = model.predict(
        source="sample_images/test.jpg",   # image or video path
        conf=0.25,
        save=True
    )

if __name__ == "__main__":
    main()
