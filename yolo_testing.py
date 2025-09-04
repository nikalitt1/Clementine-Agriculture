import pyzed.sl as sl
import cv2
import signal
from ultralytics import YOLO

stop = False

def signal_handler(sig, frame):
    global stop
    stop = True

signal.signal(signal.SIGINT, signal_handler)

def main():
    model_path = "belt.engine"
    model = YOLO(model_path)

    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD1080
    init_params.camera_fps = 30

    if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
        print("Failed to open ZED camera")
        return

    runtime_params = sl.RuntimeParameters()
    frame_width = 1920
    frame_height = 1080

    output_file = "zed_yolo_detected.avi"
    out = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*"XVID"), 30, (frame_width, frame_height))
    if not out.isOpened():
        print("Failed to open VideoWriter")
        zed.close()
        return

    image = sl.Mat()
    print("Detection started. Press Ctrl+C or 'q' to stop.")

    confidence_threshold = 0.4  # Set your threshold here

    while not stop:
        if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
            zed.retrieve_image(image, sl.VIEW.LEFT)
            frame = image.get_data()
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

            results = model(frame_bgr)

            boxes = results[0].boxes  # Boxes object
            filtered = [box for box in boxes if box.conf >= confidence_threshold]

            annotated_frame = frame_bgr.copy()
            for box in filtered:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0]
                cls = int(box.cls[0])
                class_name = model.names[cls]

                label = f"{class_name} {conf:.2f}"

                if class_name == "weed":
                    box_color = (0, 0, 255)       # Red (BGR)
                    text_color = (0, 0, 255)
                else:
                    box_color = (0, 255, 0)       # Green (BGR)
                    text_color = (0, 255, 0)

                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                radius = max((x2 - x1), (y2 - y1)) // 2

                # Draw bounding circle
                cv2.circle(annotated_frame, (cx, cy), radius, box_color, 2)

                # Draw red cross inside the circle for mock_weed
                if class_name == "weed":
                    line_length = radius // 2
                    cv2.line(annotated_frame, (cx - line_length, cy), (cx + line_length, cy), box_color, 2)
                    cv2.line(annotated_frame, (cx, cy - line_length), (cx, cy + line_length), box_color, 2)

                cv2.putText(annotated_frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)

            out.write(annotated_frame)
            preview = cv2.resize(annotated_frame, (640, 360))
            cv2.imshow("YOLOv11n ZED Detection", preview)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            print("Grab failed")

    print("Stopping...")
    out.release()
    zed.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
