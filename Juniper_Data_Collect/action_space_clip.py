#!/usr/bin/env python3
"""
filter_and_visualize_action_spaces.py
Filters overlapping action spaces between arm groups and displays
the new filtered action spaces on the ZED camera feed.
"""
import pandas as pd
import os
import cv2
import pyzed.sl as sl
import time
import sys

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
CSV_DIR = "/home/nika/Juniper1_data"
CSV_FILES = {
    "arm1": os.path.join(CSV_DIR, "J1_A1_data.csv"),
    "arm2": os.path.join(CSV_DIR, "J1_A2_data.csv"),
    "arm3": os.path.join(CSV_DIR, "J1_A3_data.csv"),
    "arm4": os.path.join(CSV_DIR, "J1_A4_data.csv"),
}

# Color scheme for each arm (BGR format for OpenCV)
ARM_COLORS = {
    "arm1": (255, 0, 0),      # Blue
    "arm2": (0, 255, 0),      # Green
    "arm3": (0, 0, 255),      # Red
    "arm4": (255, 255, 0),    # Cyan
}

# ─────────────────────────────────────────────
# Load and process data
# ─────────────────────────────────────────────
def load_csv(csv_path, arm_name):
    """Load a CSV and ensure numeric columns."""
    try:
        df = pd.read_csv(csv_path)
        for col in ["Pixel_X", "Pixel_Y"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["Pixel_X", "Pixel_Y"])
        print(f"✅ Loaded {arm_name}: {len(df)} rows")
        return df
    except Exception as e:
        print(f"❌ Error loading {arm_name}: {e}")
        return None

def get_action_space(df):
    """Get the action space bounding box for a dataframe."""
    if df.empty:
        return None
    return {
        "x_min": float(df["Pixel_X"].min()),
        "x_max": float(df["Pixel_X"].max()),
        "y_min": float(df["Pixel_Y"].min()),
        "y_max": float(df["Pixel_Y"].max()),
    }

def boxes_overlap(box1, box2):
    """Check if two bounding boxes overlap."""
    if box1 is None or box2 is None:
        return False
    
    if box1["x_max"] < box2["x_min"] or box2["x_max"] < box1["x_min"]:
        return False
    if box1["y_max"] < box2["y_min"] or box2["y_max"] < box1["y_min"]:
        return False
    
    return True

def filter_and_save_data():
    """Filter overlapping data and save CSVs."""
    print("=" * 60)
    print("FILTERING OVERLAPPING ACTION SPACES")
    print("=" * 60)
    
    # Load all CSVs
    print("\n📂 Loading CSV files...")
    dfs = {}
    for arm_key, csv_path in CSV_FILES.items():
        dfs[arm_key] = load_csv(csv_path, arm_key)
    
    # Check if all loaded
    if any(df is None for df in dfs.values()):
        print("❌ Failed to load all CSVs. Exiting.")
        return None
    
    # Group 1: Arms 1 & 3
    print("\n" + "=" * 60)
    print("PROCESSING GROUP 1 (Arms 1 & 3)")
    print("=" * 60)
    arm1_data = dfs["arm1"].copy()
    arm3_data = dfs["arm3"].copy()
    df_group1 = pd.concat([arm1_data, arm3_data], ignore_index=True)
    print(f"Combined {len(arm1_data)} + {len(arm3_data)} = {len(df_group1)} rows")
    
    # Group 2: Arms 2 & 4
    print("\n" + "=" * 60)
    print("PROCESSING GROUP 2 (Arms 2 & 4)")
    print("=" * 60)
    arm2_data = dfs["arm2"].copy()
    arm4_data = dfs["arm4"].copy()
    df_group2 = pd.concat([arm2_data, arm4_data], ignore_index=True)
    print(f"Combined {len(arm2_data)} + {len(arm4_data)} = {len(df_group2)} rows")
    
    # Get bounding boxes
    print("\n" + "=" * 60)
    print("CHECKING FOR OVERLAPS")
    print("=" * 60)
    
    box_group1 = get_action_space(df_group1)
    print(f"Group 1 action space: X=[{box_group1['x_min']:.1f}, {box_group1['x_max']:.1f}] "
          f"Y=[{box_group1['y_min']:.1f}, {box_group1['y_max']:.1f}]")
    
    box_group2 = get_action_space(df_group2)
    print(f"Group 2 action space: X=[{box_group2['x_min']:.1f}, {box_group2['x_max']:.1f}] "
          f"Y=[{box_group2['y_min']:.1f}, {box_group2['y_max']:.1f}]")
    
    # Check for overlap
    if boxes_overlap(box_group1, box_group2):
        print("⚠️  Bounding boxes overlap! Filtering...")
    else:
        print("✅ No overlap between groups")
    
    # Filter arm2 and arm4 individually
    print("\n" + "=" * 60)
    print("FILTERING INDIVIDUAL ARMS")
    print("=" * 60)
    
    arm2_original_count = len(arm2_data)
    arm2_filtered = arm2_data[
        ~(arm2_data["Pixel_X"].between(box_group1["x_min"], box_group1["x_max"]) &
          arm2_data["Pixel_Y"].between(box_group1["y_min"], box_group1["y_max"]))
    ].copy()
    arm2_removed = arm2_original_count - len(arm2_filtered)
    
    arm4_original_count = len(arm4_data)
    arm4_filtered = arm4_data[
        ~(arm4_data["Pixel_X"].between(box_group1["x_min"], box_group1["x_max"]) &
          arm4_data["Pixel_Y"].between(box_group1["y_min"], box_group1["y_max"]))
    ].copy()
    arm4_removed = arm4_original_count - len(arm4_filtered)
    
    print(f"✅ Arm 1: {len(arm1_data)} rows (unchanged)")
    print(f"✅ Arm 3: {len(arm3_data)} rows (unchanged)")
    print(f"✅ Arm 2: {arm2_original_count} → {len(arm2_filtered)} rows ({arm2_removed} removed)")
    print(f"✅ Arm 4: {arm4_original_count} → {len(arm4_filtered)} rows ({arm4_removed} removed)")
    
    # Save filtered CSVs
    print("\n" + "=" * 60)
    print("SAVING FILTERED CSV FILES")
    print("=" * 60)
    
    output_files = {
        "arm1": os.path.join(CSV_DIR, "J1_A1_data_filtered.csv"),
        "arm2": os.path.join(CSV_DIR, "J1_A2_data_filtered.csv"),
        "arm3": os.path.join(CSV_DIR, "J1_A3_data_filtered.csv"),
        "arm4": os.path.join(CSV_DIR, "J1_A4_data_filtered.csv"),
    }
    
    # Remove Pitch_deg and Roll_deg columns if they exist
    columns_to_drop = ["Pitch_deg", "Roll_deg"]
    
    for col in columns_to_drop:
        if col in arm1_data.columns:
            arm1_data = arm1_data.drop(columns=[col])
        if col in arm2_filtered.columns:
            arm2_filtered = arm2_filtered.drop(columns=[col])
        if col in arm3_data.columns:
            arm3_data = arm3_data.drop(columns=[col])
        if col in arm4_filtered.columns:
            arm4_filtered = arm4_filtered.drop(columns=[col])
    
    arm1_data.to_csv(output_files["arm1"], index=False)
    print(f"✅ Saved {output_files['arm1']} ({len(arm1_data)} rows)")
    
    arm2_filtered.to_csv(output_files["arm2"], index=False)
    print(f"✅ Saved {output_files['arm2']} ({len(arm2_filtered)} rows)")
    
    arm3_data.to_csv(output_files["arm3"], index=False)
    print(f"✅ Saved {output_files['arm3']} ({len(arm3_data)} rows)")
    
    arm4_filtered.to_csv(output_files["arm4"], index=False)
    print(f"✅ Saved {output_files['arm4']} ({len(arm4_filtered)} rows)")
    
    # Return filtered action spaces for visualization
    filtered_action_spaces = {
        "arm1": get_action_space(arm1_data),
        "arm2": get_action_space(arm2_filtered),
        "arm3": get_action_space(arm3_data),
        "arm4": get_action_space(arm4_filtered),
    }
    
    return filtered_action_spaces

def visualize_on_camera(action_spaces):
    """Visualize filtered action spaces on ZED camera feed."""
    print("\n" + "=" * 60)
    print("INITIALIZING CAMERA")
    print("=" * 60)
    
    # Wait before opening camera
    print("Waiting 3 seconds for system to stabilize...")
    for i in range(3, 0, -1):
        print(f"{i}...", end=" ", flush=True)
        time.sleep(1)
    print("Starting camera!\n")
    
    # Setup ZED
    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD1080
    init_params.camera_fps = 30
    
    print("Opening ZED camera...")
    sys.stdout.flush()
    
    result = zed.open(init_params)
    
    if result != sl.ERROR_CODE.SUCCESS:
        print(f"❌ Failed to open ZED camera: {result}")
        sys.stdout.flush()
        return
    
    runtime_params = sl.RuntimeParameters()
    image = sl.Mat()
    
    print(f"✅ ZED camera opened successfully!")
    print("Press 'q' to quit, 's' to save frame\n")
    sys.stdout.flush()
    
    # CSV resolution
    csv_width = 1920
    csv_height = 1080
    
    frame_count = 0
    
    while True:
        if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
            zed.retrieve_image(image, sl.VIEW.LEFT)
            frame = image.get_data()
            
            # Convert BGRA to BGR
            draw = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            
            fh, fw = draw.shape[:2]
            
            # Scale factors
            scale_x = fw / csv_width
            scale_y = fh / csv_height
            
            # Draw action space bounding box for each arm
            for arm_key, action_space in action_spaces.items():
                if action_space is None:
                    continue
                
                # Get pixel coordinates from action space
                x_min_csv = action_space["x_min"]
                x_max_csv = action_space["x_max"]
                y_min_csv = action_space["y_min"]
                y_max_csv = action_space["y_max"]
                
                # Convert to video frame coordinates
                x1 = max(0, min(fw - 1, int(x_min_csv * scale_x)))
                y1 = max(0, min(fh - 1, int(y_min_csv * scale_y)))
                x2 = max(0, min(fw - 1, int(x_max_csv * scale_x)))
                y2 = max(0, min(fh - 1, int(y_max_csv * scale_y)))
                
                # Get arm color
                color = ARM_COLORS.get(arm_key, (255, 255, 255))
                
                # Draw semi-transparent filled rectangle
                overlay = draw.copy()
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
                cv2.addWeighted(overlay, 0.15, draw, 0.85, 0, draw)
                
                # Draw solid border
                cv2.rectangle(draw, (x1, y1), (x2, y2), color, 3, cv2.LINE_AA)
                
                # Add label
                label = f"{arm_key.upper()}"
                text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                text_x = x1 + 10
                text_y = y1 + 25
                
                # Background for text
                cv2.rectangle(
                    draw,
                    (text_x - 5, text_y - text_size[1] - 5),
                    (text_x + text_size[0] + 5, text_y + 5),
                    color,
                    -1
                )
                cv2.putText(
                    draw,
                    label,
                    (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA
                )
                
                # Draw corners
                corner_len = 20
                cv2.line(draw, (x1, y1), (x1 + corner_len, y1), color, 3)
                cv2.line(draw, (x1, y1), (x1, y1 + corner_len), color, 3)
                cv2.line(draw, (x2, y1), (x2 - corner_len, y1), color, 3)
                cv2.line(draw, (x2, y1), (x2, y1 + corner_len), color, 3)
                cv2.line(draw, (x1, y2), (x1 + corner_len, y2), color, 3)
                cv2.line(draw, (x1, y2), (x1, y2 - corner_len), color, 3)
                cv2.line(draw, (x2, y2), (x2 - corner_len, y2), color, 3)
                cv2.line(draw, (x2, y2), (x2, y2 - corner_len), color, 3)
            
            # Add title
            title = "FILTERED ACTION SPACES"
            title_size = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
            title_x = (fw - title_size[0]) // 2
            cv2.putText(
                draw,
                title,
                (title_x, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2,
                cv2.LINE_AA
            )
            
            # Add legend
            legend_y = fh - 100
            legend_x = 20
            cv2.putText(
                draw,
                "Filtered Legend:",
                (legend_x, legend_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (200, 200, 200),
                1,
                cv2.LINE_AA
            )
            
            for i, (arm_key, _) in enumerate(action_spaces.items()):
                color = ARM_COLORS.get(arm_key, (255, 255, 255))
                y_offset = legend_y + 25 + (i * 20)
                cv2.circle(draw, (legend_x + 10, y_offset), 5, color, -1)
                cv2.putText(
                    draw,
                    f"{arm_key.upper()}",
                    (legend_x + 25, y_offset + 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    1,
                    cv2.LINE_AA
                )
            
            # Resize for display
            display_frame = cv2.resize(draw, (640, 360))
            
            # Display
            cv2.imshow("Filtered Action Spaces - ZED", display_frame)
            
            frame_count += 1
            if frame_count == 1:
                print("✅ Camera feed is active!")
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Quitting...")
                break
            elif key == ord('s'):
                filename = f"filtered_action_space_frame_{int(time.time())}.png"
                cv2.imwrite(filename, draw)
                print(f"Saved frame to {filename}")
        else:
            print("⚠️ Grab failed, retrying...")
            time.sleep(0.1)
    
    # Cleanup
    zed.close()
    cv2.destroyAllWindows()
    print("✅ Done!")

def main():
    # Filter data and get action spaces
    action_spaces = filter_and_save_data()
    
    if action_spaces is None:
        print("Filtering failed. Exiting.")
        return
    
    # Visualize on camera
    visualize_on_camera(action_spaces)

if __name__ == "__main__":
    main()
