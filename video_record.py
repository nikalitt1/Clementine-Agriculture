import pyzed.sl as sl
import cv2
import signal

stop = False

def signal_handler(sig, frame):
    global stop
    stop = True

signal.signal(signal.SIGINT, signal_handler)

def main():
    # Create ZED Camera object
    zed = sl.Camera()

    # Init parameters
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD1080  # or sl.RESOLUTION.VGA for smaller
    init_params.camera_fps = 30

    # Open camera
    if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
        print("Failed to open ZED camera")
        return

    runtime_params = sl.RuntimeParameters()

    # Set frame size for VideoWriter (must match camera_resolution)
    frame_width = 1920
    frame_height = 1080

    output_file = "belt_training20.avi"
    fourcc = cv2.VideoWriter_fourcc(*"XVID")  # Use XVID codec

    out = cv2.VideoWriter(output_file, fourcc, 60, (frame_width, frame_height))

    if not out.isOpened():
        print("Failed to open VideoWriter with XVID codec.")
        zed.close()
        return

    image = sl.Mat()

    print("Recording started. Press Ctrl+C or 'q' to stop.")

    while not stop:
        if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
            zed.retrieve_image(image, sl.VIEW.LEFT)
            frame = image.get_data()

            # Convert BGRA to BGR (drop alpha)
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

            # Resize if needed (should already match, but just in case)
            if (frame_bgr.shape[1], frame_bgr.shape[0]) != (frame_width, frame_height):
                frame_resized = cv2.resize(frame_bgr, (frame_width, frame_height))
            else:
                frame_resized = frame_bgr

            out.write(frame_resized)

            preview = cv2.resize(frame_bgr, (640, 360))
            cv2.imshow("ZED X Mini Preview", preview)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            print("Grab failed")

    print("Stopping recording...")
    out.release()
    zed.close()
    cv2.destroyAllWindows()
    print(f"Video saved to {output_file}")

if __name__ == "__main__":
    main()
