import serial
import time
import glob
from smbus2 import SMBus
import threading
import cv2
import signal
import pyzed.sl as sl

# === PCA9685 Constants & I2C bus setup ===
PCA9685_ADDRESS = 0x40
MODE1 = 0x00
MODE2 = 0x01
LED0_ON_L = 0x06
LED0_ON_H = 0x07
LED0_OFF_L = 0x08
LED0_OFF_H = 0x09

I2C_BUS_NUMBER = 7
bus = SMBus(I2C_BUS_NUMBER)

stop = False

def signal_handler(sig, frame):
    global stop
    stop = True

signal.signal(signal.SIGINT, signal_handler)

def init_pca9685():
    bus.write_byte_data(PCA9685_ADDRESS, MODE1, 0x00)
    bus.write_byte_data(PCA9685_ADDRESS, MODE2, 0x04)
    time.sleep(0.1)

def set_pwm(channel, pulse_width):
    on_time = 0
    off_time = int(pulse_width)
    bus.write_byte_data(PCA9685_ADDRESS, LED0_ON_L + 4 * channel, on_time & 0xFF)
    bus.write_byte_data(PCA9685_ADDRESS, LED0_ON_H + 4 * channel, (on_time >> 8) & 0xFF)
    bus.write_byte_data(PCA9685_ADDRESS, LED0_OFF_L + 4 * channel, off_time & 0xFF)
    bus.write_byte_data(PCA9685_ADDRESS, LED0_OFF_H + 4 * channel, (off_time >> 8) & 0xFF)

def set_duty_cycle(channel, percent):
    percent = max(0, min(100, percent))
    ticks = int(percent * 4095 / 100)
    set_pwm(channel, ticks)

def list_serial_ports():
    return glob.glob('/dev/ttyUSB*')

def checksum(data):
    return (~sum(data)) & 0xFF

def move_servo(ser, servo_id, position):
    pos_val = int((position / 360.0) * 4095)
    pos_l = pos_val & 0xFF
    pos_h = (pos_val >> 8) & 0xFF
    packet = [0xFF, 0xFF, servo_id, 7, 0x03, 0x2A, pos_l, pos_h, 0x00, 0x00]
    packet.append(checksum(packet[2:]))
    ser.write(bytearray(packet))
    time.sleep(0.05)

def scan_servos(ser):
    found_ids = []
    for servo_id in range(1, 21):
        packet = [0xFF, 0xFF, servo_id, 2, 0x01]
        packet.append(checksum(packet[2:]))
        ser.write(bytearray(packet))
        time.sleep(0.03)
        if ser.in_waiting > 0:
            response = ser.read(ser.in_waiting)
            if len(response) >= 6 and response[0] == 0xFF and response[1] == 0xFF:
                found_ids.append(servo_id)
    return found_ids

# === Input Command Thread ===
def control_thread(servo_to_port, port_connections):
    global stop
    print("\n🎮 Control servos and PCA9685 PWM:")
    print("  Format: `servo_id angle pwm_channel percent`")
    print("  Type `q` to quit.\n")
    while not stop:
        try:
            user_input = input("Enter command: ").strip()
            if user_input.lower() == 'q':
                stop = True
                break

            tokens = user_input.split()
            if len(tokens) != 4:
                print("❌ Format: servo_id angle pwm_channel percent")
                continue

            sid = int(tokens[0])
            angle = float(tokens[1])
            pwm_channel = int(tokens[2])
            pwm_percent = float(tokens[3])

            if sid not in servo_to_port:
                print(f"❌ Servo ID {sid} not found.")
            elif not (0 <= angle <= 360):
                print("❌ Angle must be 0–360.")
            else:
                port = servo_to_port[sid]
                ser = port_connections.get(port)
                if ser:
                    move_servo(ser, sid, angle)
                    print(f"✅ Moved servo {sid} to {angle}°")

            if not (0 <= pwm_channel <= 15):
                print("❌ PWM channel must be 0–15.")
            elif not (0 <= pwm_percent <= 100):
                print("❌ PWM % must be 0–100.")
            else:
                set_duty_cycle(pwm_channel, pwm_percent)
                print(f"✅ Set PWM {pwm_channel} to {pwm_percent}%")

        except Exception as e:
            print(f"❌ Error: {e}")

# === Main Function ===
def main():
    global stop

    # Setup ZED
    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD1080
    init_params.camera_fps = 30

    if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
        print("❌ Failed to open ZED camera")
        return

    runtime_params = sl.RuntimeParameters()
    image = sl.Mat()

    # Init PCA9685
    init_pca9685()

    # Setup servos
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
            if ids:
                print(f"✅ Found servos on {port}: {ids}")
                for sid in ids:
                    servo_to_port[sid] = port
            else:
                print(f"🚫 No servos on {port}")
        except Exception as e:
            print(f"❌ Error opening {port}: {e}")

    if not servo_to_port:
        print("No servos found. Exiting.")
        return

    port_connections = {}
    for port in set(servo_to_port.values()):
        try:
            port_connections[port] = serial.Serial(port, baudrate=baud, timeout=0.1)
        except Exception as e:
            print(f"❌ Couldn't open {port}: {e}")

    # Start command thread
    threading.Thread(target=control_thread, args=(servo_to_port, port_connections), daemon=True).start()

    print("📷 ZED Preview started. Press 'q' or Ctrl+C to quit.")

    # ZED Live Preview Loop (main thread)
    while not stop:
        if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
            zed.retrieve_image(image, sl.VIEW.LEFT)
            frame = image.get_data()
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            preview = cv2.resize(frame_bgr, (640, 360))

            # === Brightest spot detection ===
            gray = cv2.cvtColor(preview, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (15, 15), 0)
            _, thresh = cv2.threshold(blurred, 215, 255, cv2.THRESH_BINARY)

            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                if cv2.contourArea(largest_contour) > 100:
                    x, y, w, h = cv2.boundingRect(largest_contour)
                    cv2.rectangle(preview, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(preview, "Bright Spot", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, (0, 255, 0), 2)

            cv2.imshow("ZED X Mini Preview", preview)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                stop = True
                break
        else:
            print("⚠️ Grab failed")

    # Cleanup
    for ser in port_connections.values():
        ser.close()
    bus.close()
    zed.close()
    cv2.destroyAllWindows()
    print("🔚 Exiting.")

if __name__ == "__main__":
    main()

