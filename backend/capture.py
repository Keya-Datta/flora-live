import sys
import cv2
from ultralytics import YOLO


def main():
    print("Python executable:", sys.executable)

    print(" Loading YOLO model (this can take a moment the first time)...")
    model = YOLO("yolov8n.pt")
    print(" Model loaded.")

    print(" Opening webcam...")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print(" Could not open webcam. Try changing the index (0 → 1 or 2).")
        return

    print("Webcam opened. A window should appear. Press 'q' to quit.")

    # Explicitly create a window (helps on macOS)
    cv2.namedWindow("FLORA-LIVE – YOLO Webcam", cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        # Run YOLO (no console spam)
        results = model(frame, verbose=False)

        # Use ultralytics' built-in plotting to draw detections
        annotated_frame = results[0].plot()

        # Show the frame
        cv2.imshow("FLORA-LIVE – YOLO Webcam", annotated_frame)

        # Check for 'q' key to quit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print(" Quit key pressed, exiting.")
            break

    cap.release()
    cv2.destroyAllWindows()
    print(" Cleaned up and closed.")


if __name__ == "__main__":
    main()