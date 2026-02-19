import serial
import time
import glob
from smbus2 import SMBus
import cv2
import signal
import threading
import pyzed.sl as sl
import math
from queue import Queue
import csv
from datetime import datetime
from threading import Lock
import numpy as np
from collections import deque

# === Added histories for smoothing bounding box size ===
center_history = deque(maxlen=8)  # smoothing window for center position
width_history = deque(maxlen=8)   # smoothing window for bounding box width
height_history = deque(maxlen=8)  # smoothing window for bounding box height

camera_data = {
    "pixel_x": None,
    "pixel_y": None,
    "depth": None,
    "pitch": None,
    "roll": None
}
camera_data_lock = Lock()

# === PCA9685 Constants & I2C bus setup ===
PCA9685_ADDRESS = 0x40
MODE1 = 0x00
MODE2 = 0x01
LED0_ON_L = 0x06
LED0_ON_H = 0x07
LED0_OFF_L = 0x08
LED0_OFF_H = 0x09

I2C_BUS_NUMBER = 7
bus = None

stop = False
serial_lock = threading.Lock()
frame_queue = Queue(maxsize=3)  # <-- Added for thread-safe preview frames

# === Video recording settings ===
VIDEO_OUTPUT_PATH = "j1_v5.avi"
VIDEO_FPS = 30
VIDEO_RESOLUTION = (640, 360)  # Display resolution — lower for smaller file size

# === Shadow enhancement settings ===
# SHADOW_GAMMA: < 1.0 brightens shadows (e.g. 0.5 = strong lift, 0.8 = subtle lift)
#               > 1.0 darkens image. 1.0 = no change.
# SHADOW_LIFT:  Adds a flat brightness floor to dark pixels (0–255). 0 = off.
# SHADOW_THRESHOLD: Pixels below this brightness level are considered "shadows" (0–255).
#                   Only these pixels get the lift applied. 255 = affect everything.
SHADOW_GAMMA     = 0.6   # Gamma for shadow brightening
SHADOW_LIFT      = 20    # Flat lift added to shadow pixels
SHADOW_THRESHOLD = 80    # Pixel brightness below this value is treated as shadow

def build_shadow_lut(gamma=SHADOW_GAMMA, lift=SHADOW_LIFT, threshold=SHADOW_THRESHOLD):
    """
    Build a 256-entry lookup table that:
      - Applies gamma correction + lift to pixels below `threshold`
      - Leaves pixels above `threshold` unchanged
    This is computed once and reused every frame for efficiency.
    """
    lut = np.arange(256, dtype=np.float32)
    # Gamma curve applied to shadow range, scaled back to [0, threshold]
    shadow_mask = lut < threshold
    lut[shadow_mask] = (lut[shadow_mask] / threshold) ** gamma * threshold + lift
    lut = np.clip(lut, 0, 255).astype(np.uint8)
    return lut

# Pre-build the LUT at startup
SHADOW_LUT = build_shadow_lut()

def apply_shadow_enhancement(frame_bgr):
    """Apply shadow brightening to a BGR frame using the pre-built LUT."""
    return cv2.LUT(frame_bgr, SHADOW_LUT)

def signal_handler(sig, frame):
    global stop
    stop = True

signal.signal(signal.SIGINT, signal_handler)

# === PCA9685 functions ===
def init_pca9685():
    try:
        bus.write_byte_data(PCA9685_ADDRESS, MODE1, 0x00)
        bus.write_byte_data(PCA9685_ADDRESS, MODE2, 0x04)
        time.sleep(0.1)
    except Exception as e:
        print(f"Error initializing PCA9685: {e}")

def set_pwm(channel, pulse_width):
    try:
        on_time = 0
        off_time = int(pulse_width)
        bus.write_byte_data(PCA9685_ADDRESS, LED0_ON_L + 4 * channel, on_time & 0xFF)
        bus.write_byte_data(PCA9685_ADDRESS, LED0_ON_H + 4 * channel, (on_time >> 8) & 0xFF)
        bus.write_byte_data(PCA9685_ADDRESS, LED0_OFF_L + 4 * channel, off_time & 0xFF)
        bus.write_byte_data(PCA9685_ADDRESS, LED0_OFF_H + 4 * channel, (off_time >> 8) & 0xFF)
    except Exception as e:
        print(f"Error setting PWM: {e}")
        
        
def set_duty_cycle():
    channel = 3
    percent = 0.0 # example value, adjust as needed
    percent = max(0, min(100, percent))
    ticks = int(percent * 4095 / 100)
    set_pwm(channel, ticks)

# === Servo functions ===
def list_serial_ports():
    return glob.glob('/dev/ttyUSB*')

def checksum(data):
    return (~sum(data)) & 0xFF

def move_servo(ser, servo_id, position):
    try:
        pos_val = int((position / 360.0) * 4095)
        pos_l = pos_val & 0xFF
        pos_h = (pos_val >> 8) & 0xFF
        packet = [0xFF, 0xFF, servo_id, 7, 0x03, 0x2A, pos_l, pos_h, 0x00, 0x00]
        packet.append(checksum(packet[2:]))
        with serial_lock:
            ser.write(bytearray(packet))
        print(f"Sent to servo {servo_id}: pos={position}, packet={[hex(b) for b in packet]}")
        time.sleep(0.05)
    except Exception as e:
        print(f"Error moving servo {servo_id}: {e}")

def scan_servos(ser):
    found_ids = []
    try:
        for servo_id in range(1, 21):
            packet = [0xFF, 0xFF, servo_id, 2, 0x01]
            packet.append(checksum(packet[2:]))
            with serial_lock:
                ser.write(bytearray(packet))
            time.sleep(0.03)
            if ser.in_waiting > 0:
                response = ser.read(ser.in_waiting)
                if len(response) >= 6 and response[0] == 0xFF and response[1] == 0xFF:
                    found_ids.append(servo_id)
    except Exception as e:
        print(f"Error scanning servos: {e}")
    return found_ids

def generate_zigzag_2d(x_start, x_end, x_step, y_start, y_end, y_step):
    import numpy as np

    if x_start < x_end:
        x_vals = np.arange(x_start, x_end + x_step, x_step)
    else:
        x_vals = np.arange(x_start, x_end - x_step, -x_step)  # negative step when start > end

    if y_start < y_end:
        y_vals = np.arange(y_start, y_end + y_step, y_step)
    else:
        y_vals = np.arange(y_start, y_end - y_step, -y_step)

    direction = 1
    while True:
        for y in y_vals:
            if direction == 1:
                for x in x_vals:
                    yield (x, y)
            else:
                for x in reversed(x_vals):
                    yield (x, y)
            direction *= -1

# === Threads ===
def camera_thread_func(zed):
    runtime_params = sl.RuntimeParameters()
    image = sl.Mat()
    depth = sl.Mat()
    sensors_data = sl.SensorsData()

    # === Initialize VideoWriter ===
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_writer = cv2.VideoWriter(VIDEO_OUTPUT_PATH, fourcc, VIDEO_FPS, VIDEO_RESOLUTION)
    if not video_writer.isOpened():
        print("❌ Failed to open VideoWriter. Recording will be skipped.")
        video_writer = None
    else:
        print(f"🎥 Recording to {VIDEO_OUTPUT_PATH} at {VIDEO_FPS} FPS, resolution {VIDEO_RESOLUTION}")

    global stop
    while not stop:
        if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
            zed.retrieve_image(image, sl.VIEW.LEFT)
            frame = image.get_data()
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

            # === Apply shadow brightening ===
            frame_bgr = apply_shadow_enhancement(frame_bgr)

            zed.retrieve_measure(depth, sl.MEASURE.DEPTH)

            # === Blue Region Detection ===
            hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
            
            # Define HSV range for blue (adjust if needed)
            lower_blue = np.array([90, 150, 50])   # H, S, V lower bound
            upper_blue = np.array([130, 255, 255])  # H, S, V upper bound
            
            mask = cv2.inRange(hsv, lower_blue, upper_blue)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))
            mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, np.ones((5,5), np.uint8))
            
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            x = y = None
            if contours:
                # Pick the largest blue contour
                largest = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(largest)

                if area > 40:
                    x_rect, y_rect, w, h = cv2.boundingRect(largest)
                    cx = x_rect + w // 2
                    cy = y_rect + h // 2

                    # Update histories for smoothing
                    center_history.append((cx, cy))
                    width_history.append(w)
                    height_history.append(h)

                    # Average over histories for smoothness
                    avg_cx = int(np.mean([p[0] for p in center_history]))
                    avg_cy = int(np.mean([p[1] for p in center_history]))
                    avg_w = int(np.mean(width_history))
                    avg_h = int(np.mean(height_history))

                    x, y = avg_cx, avg_cy

                    # Draw rectangle and circle with smoothed values
                    top_left = (avg_cx - avg_w // 2, avg_cy - avg_h // 2)
                    bottom_right = (avg_cx + avg_w // 2, avg_cy + avg_h // 2)

                    cv2.rectangle(frame_bgr, top_left, bottom_right, (255, 0, 0), 2)
                    cv2.circle(frame_bgr, (avg_cx, avg_cy), 5, (0, 0, 255), -1)
                    cv2.putText(frame_bgr, "Blue Region", (top_left[0], top_left[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

            if x is not None and y is not None:
                depth_err, depth_value = depth.get_value(x, y)
                if depth_err == sl.ERROR_CODE.SUCCESS:
                    with camera_data_lock:
                        camera_data["pixel_x"] = x
                        camera_data["pixel_y"] = y
                        camera_data["depth"] = round(depth_value / 1000.0, 3)  # mm → meters
                else:
                    print("⚠️ Depth not available")
            else:
                print("⚠️ No bright spot")

            if zed.get_sensors_data(sensors_data, sl.TIME_REFERENCE.CURRENT) == sl.ERROR_CODE.SUCCESS:
                imu = sensors_data.get_imu_data().get_pose().get_orientation().get()
                w, x, y, z = imu
                t0 = +2.0 * (w * x + y * z)
                t1 = +1.0 - 2.0 * (x * x + y * y)
                roll = math.degrees(math.atan2(t0, t1))

                t2 = +2.0 * (w * y - z * x)
                t2 = max(min(t2, +1.0), -1.0)
                pitch = math.degrees(math.asin(t2))
                with camera_data_lock:
                    camera_data["pitch"] = round(pitch, 2)
                    camera_data["roll"] = round(roll, 2)

            # === Write annotated frame to video file ===
            if video_writer is not None:
                # Ensure frame matches the declared VIDEO_RESOLUTION before writing
                if (frame_bgr.shape[1], frame_bgr.shape[0]) != VIDEO_RESOLUTION:
                    frame_to_write = cv2.resize(frame_bgr, VIDEO_RESOLUTION)
                else:
                    frame_to_write = frame_bgr
                video_writer.write(frame_to_write)

            # Send frame to main thread for preview
            if not frame_queue.full():
                frame_queue.put(frame_bgr)

        else:
            print("⚠️ Grab failed")
            time.sleep(0.5)

    # === Release VideoWriter on thread exit ===
    if video_writer is not None:
        video_writer.release()
        print(f"✅ Video saved to {VIDEO_OUTPUT_PATH}")

    print("Camera thread exiting.")

def servo_thread_func(ser_x, ser_y, servo_x_id, servo_y_id):
    x_start, x_end = 160, 190
    y_start, y_end = 205, 260
    zigzag_gen = generate_zigzag_2d(x_start, x_end, 0.5, y_start, y_end, 0.5)

    log_filename = "J1_A5_data3.csv"
    with open(log_filename, mode='w', newline='') as logfile:
        writer = csv.writer(logfile)
        writer.writerow(["Motor_X_Angle", "Motor_Y_Angle", "Pixel_X", "Pixel_Y", "Depth_m", "Pitch_deg", "Roll_deg"])

        global stop
        while not stop:
            try:
                servo_x_angle, servo_y_angle = next(zigzag_gen)
                if ser_x and ser_y:
                    move_servo(ser_x, servo_x_id, servo_x_angle)
                    move_servo(ser_y, servo_y_id, servo_y_angle)

                with camera_data_lock:
                    px = camera_data["pixel_x"]
                    py = camera_data["pixel_y"]
                    d = camera_data["depth"]
                    pitch = camera_data["pitch"]
                    roll = camera_data["roll"]

                timestamp = datetime.now().isoformat()
                writer.writerow([servo_x_angle, servo_y_angle, px, py, d, pitch, roll])
                logfile.flush()

                time.sleep(0.1)

            except Exception as e:
                print(f"Servo thread error: {e}")
                break
    print("Servo thread exiting.")


# === Main Entry ===
def main():
    global stop, bus
    baud = 1000000
    ports = list_serial_ports()
    servo_to_port = {}

    print(f"🧭 Found ports: {ports}")
    for port in ports:
        try:
            ser = serial.Serial(port, baudrate=baud, timeout=0.1)
            time.sleep(0.2)
            ids = scan_servos(ser)
            ser.close()
            for sid in ids:
                servo_to_port[sid] = port
        except Exception as e:
            print(f"Error scanning {port}: {e}")

    print("Opening ZED camera...")
    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD1080  # Keep 1080p capture
    init_params.camera_fps = 30
    if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
        print("❌ Failed to open ZED camera")
        return

    tracking_params = sl.PositionalTrackingParameters()
    zed.enable_positional_tracking(tracking_params)

    try:
        bus = SMBus(I2C_BUS_NUMBER)
        init_pca9685()
        set_duty_cycle()

    except Exception as e:
        print(f"Failed to open I2C: {e}")
        zed.close()
        return

    port_connections = {}
    for port in set(servo_to_port.values()):
        port_connections[port] = serial.Serial(port, baudrate=baud, timeout=0.1)

    servo_x_id = 8 # Servo IDS
    servo_y_id = 7
    
    
    ser_x = port_connections.get(servo_to_port.get(servo_x_id))
    ser_y = port_connections.get(servo_to_port.get(servo_y_id))

    if ser_x and ser_y:
        move_servo(ser_x, servo_x_id, 200)
        move_servo(ser_y, servo_y_id, 200)
    else:
        print("❌ Servos not found")

    cam_thread = threading.Thread(target=camera_thread_func, args=(zed,))
    servo_thread = threading.Thread(target=servo_thread_func, args=(ser_x, ser_y, servo_x_id, servo_y_id))

    cam_thread.start()
    servo_thread.start()

    # Display frames from the queue (in main thread)
    while not stop:
        if not frame_queue.empty():
            frame = frame_queue.get()

            # Resize frame to 640x480 for display
            frame_resized = cv2.resize(frame, (640, 360))
            
            cv2.imshow("ZED X Mini Preview", frame_resized)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                stop = True
                break
        else:
            time.sleep(0.01)

    cam_thread.join()
    servo_thread.join()
    for ser in port_connections.values():
        ser.close()
    if bus:
        bus.close()
    zed.close()
    cv2.destroyAllWindows()
    print("🔚 Exiting.")

if __name__ == "__main__":
    main()
