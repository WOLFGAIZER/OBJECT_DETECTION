import cv2
import time
from ultralytics import YOLO

def run_webcam():
    model_path = r"C:\Users\sai78\Desktop\OBJECT_DETECTION\runs\FruitDetection_Results\weights\best.pt"

    print("Loading model...")
    model = YOLO(model_path)

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Webcam started. Press 'q' to quit.")

    prev_time = 0

    while True:
        success, frame = cap.read()
        if not success:
            break

        results = model.predict(source=frame, conf=0.5, save=False, stream=True)

        for r in results:
            annotated_frame = r.plot()

            curr_time = time.time()
            fps = 1 / (curr_time - prev_time)
            prev_time = curr_time

            cv2.putText(annotated_frame, f"FPS: {int(fps)}", (20, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("YOLOv8 Fruit Detection (Webcam)", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_webcam()