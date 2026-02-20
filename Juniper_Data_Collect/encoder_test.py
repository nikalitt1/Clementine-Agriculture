import serial
import time

def checksum(data):
    """Calculate Feetech checksum: ~(ID + Length + Instruction + Parameters)"""
    return (~sum(data)) & 0xFF

def read_servo_position(ser, servo_id):
    """Read current position from STS3215 servo"""
    length = 4
    instruction = 0x02  # READ command
    start_addr = 0x38   # Position register address
    read_length = 0x02  # Read 2 bytes
    
    data = [servo_id, length, instruction, start_addr, read_length]
    cs = checksum(data)
    packet = [0xFF, 0xFF] + data + [cs]
    
    ser.reset_input_buffer()
    ser.write(bytearray(packet))
    ser.flush()
    time.sleep(0.01)
    
    response = ser.read(ser.in_waiting if ser.in_waiting > 0 else 8)
    
    if len(response) < 8:
        return None
    
    if response[0] != 0xFF or response[1] != 0xFF:
        return None
    
    if response[2] != servo_id:
        return None
    
    error = response[4]
    if error != 0:
        return None
    
    # Verify checksum
    received_cs = response[-1]
    calculated_cs = checksum(list(response[2:-1]))
    if received_cs != calculated_cs:
        return None
    
    # Extract position
    pos_low = response[5]
    pos_high = response[6]
    position = (pos_high << 8) | pos_low
    angle = (position / 4095.0) * 360.0
    
    return angle, position

def ping_servo(ser, servo_id):
    """Ping servo to verify it's connected and responding"""
    data = [servo_id, 2, 0x01]  # PING instruction
    cs = checksum(data)
    packet = [0xFF, 0xFF] + data + [cs]
    
    ser.reset_input_buffer()
    ser.write(bytearray(packet))
    ser.flush()
    time.sleep(0.01)
    
    response = ser.read(6)
    return len(response) >= 6 and response[0] == 0xFF and response[1] == 0xFF

def scan_servos(ser, port_name, max_id=20):
    """Scan for active servos on a port"""
    print(f"\nScanning {port_name} for servos...")
    found = []
    for servo_id in range(1, max_id + 1):
        if ping_servo(ser, servo_id):
            print(f"  Found servo ID {servo_id}")
            found.append(servo_id)
        time.sleep(0.02)
    return found

# === Configuration ===
PORT_CONFIG = {
    '/dev/ttyUSB0': [5,6,7,8],  # Will auto-detect servos
    '/dev/ttyUSB1': [1,2,3,4]   # Will auto-detect servos
}

BAUDRATE = 1000000  # Try 1000000, 500000, or 115200 if this doesn't work

# === Initialize serial ports ===
serial_ports = {}
servo_map = {}  # Maps servo_id -> serial port object

try:
    # Open both serial ports
    for port in PORT_CONFIG.keys():
        try:
            ser = serial.Serial(
                port=port,
                baudrate=BAUDRATE,
                timeout=0.1,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE
            )
            serial_ports[port] = ser
            print(f"Opened {port} at {BAUDRATE} baud")
        except serial.SerialException as e:
            print(f"Could not open {port}: {e}")
    
    if not serial_ports:
        print("No serial ports available!")
        exit(1)
    
    # Scan for servos on each port
    print("\n" + "="*50)
    print("AUTO-DETECTING SERVOS")
    print("="*50)
    
    for port, ser in serial_ports.items():
        found_ids = scan_servos(ser, port, max_id=20)
        PORT_CONFIG[port] = found_ids
        for servo_id in found_ids:
            servo_map[servo_id] = ser
    
    # Print summary
    print("\n" + "="*50)
    print("SERVO CONFIGURATION")
    print("="*50)
    for port, ids in PORT_CONFIG.items():
        if ids:
            print(f"{port}: Servos {ids}")
        else:
            print(f"{port}: No servos found")
    
    total_servos = sum(len(ids) for ids in PORT_CONFIG.values())
    print(f"\nTotal servos detected: {total_servos}")
    
    if total_servos == 0:
        print("\nNo servos found! Check:")
        print("  1. Power supply (12V)")
        print("  2. Wiring connections")
        print("  3. Baud rate (try 500000 or 115200)")
        print("  4. Servo IDs (default is usually 1)")
        exit(1)
    
    # === Main monitoring loop ===
    print("\n" + "="*50)
    print("READING SERVO POSITIONS (Ctrl+C to stop)")
    print("="*50)
    
    while True:
        print(f"\n[{time.strftime('%H:%M:%S')}]")
        
        for port, servo_ids in PORT_CONFIG.items():
            if not servo_ids:
                continue
            
            ser = serial_ports[port]
            port_name = port.split('/')[-1]  # Get 'ttyUSB0' from '/dev/ttyUSB0'
            
            for servo_id in servo_ids:
                result = read_servo_position(ser, servo_id)
                if result is not None:
                    angle, raw_pos = result
                    print(f"  [{port_name}] Servo {servo_id:2d}: {angle:6.2f}° (raw: {raw_pos:4d})")
                else:
                    print(f"  [{port_name}] Servo {servo_id:2d}: No response")
                
                time.sleep(0.02)  # Small delay between servos
        
        time.sleep(0.5)  # Delay between full reads

except KeyboardInterrupt:
    print("\n\nStopped by user")

except Exception as e:
    print(f"\nError: {e}")
    import traceback
    traceback.print_exc()

finally:
    # Close all serial ports
    for port, ser in serial_ports.items():
        if ser.is_open:
            ser.close()
            print(f"Closed {port}")
