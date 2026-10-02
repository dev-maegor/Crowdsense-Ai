"""
================================================================================
CROWD MONITORING SYSTEM
================================================================================
Real-time crowd detection, density heatmap, and analytics visualization

Features:
  - YOLOv8 person detection from webcam feed
  - Real-time bounding boxes on detected people
  - Density heatmap showing crowd concentration areas
  - Real-time graph of people count (updated every 2 seconds)
  - Dark theme UI with high contrast visuals

Exit: Press 'q' to quit or close the window
================================================================================
"""

# ============================================================================
# IMPORTS AND DEPENDENCIES
# ============================================================================
import cv2
from ultralytics import YOLO
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import time

# ============================================================================
# MODEL & CAMERA INITIALIZATION
# ============================================================================

def read_initial_frame(capture):
    """Read a usable first frame or report a camera startup failure."""
    success, captured_frame = capture.read()
    if not success or captured_frame is None:
        raise RuntimeError("The camera opened but did not return a video frame.")
    return captured_frame

# Load pre-trained YOLOv8 nano model for person detection
model = YOLO("yolov8n.pt")

# Initialize webcam capture (0 = default camera)
cap = cv2.VideoCapture(0)

# Check if camera is accessible
if not cap.isOpened():
    print("Error: Cannot access webcam")
    cap.release()
    raise SystemExit(1)

# Get initial frame to determine video dimensions
try:
    frame = read_initial_frame(cap)
except RuntimeError as error:
    cap.release()
    cv2.destroyAllWindows()
    plt.close("all")
    raise SystemExit(f"Error: {error}") from error

frame_height, frame_width = frame.shape[:2]

# ============================================================================
# HEATMAP INITIALIZATION
# ============================================================================

# Initialize heatmap to accumulate detection points over time
# Represents crowd density - higher values = more people detected
heatmap = np.zeros((frame_height, frame_width), dtype=np.float32)

# ============================================================================
# GRAPH SETUP - Data Storage & Matplotlib Configuration
# ============================================================================

# Storage for real-time analytics graph
time_data = []          # X-axis: elapsed time in seconds
count_data = []         # Y-axis: people count values

# Timing variables - used to aggregate data per second
start_time = time.time()
last_graph_update = start_time
current_second = 0
second_counts = []      # Collects all detections within current second

# Create matplotlib figure with dark theme for visual consistency
fig, ax = plt.subplots(figsize=(12, 4), dpi=80)
fig.patch.set_facecolor('black')        # Black background
ax.set_facecolor('black')               # Black axes background

# ============================================================================
# WINDOW CONFIGURATION
# ============================================================================

# Create OpenCV display window with resizable capability
cv2.namedWindow("Crowd Monitoring System", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Crowd Monitoring System", 1920, 1080)

graph_img = None  # Placeholder for rendered graph image

# ============================================================================
# MAIN PROCESSING LOOP
# ============================================================================

while True:
    
    # ===== FRAME CAPTURE =====
    ret, frame = cap.read()
    if not ret:
        break

    # ===== YOLO PERSON DETECTION =====
    # Run YOLOv8 inference on current frame
    # class 0 = person, conf=0.25 (25% confidence threshold)
    results = model.predict(frame, imgsz=640, conf=0.25, classes=[0])

    count = 0  # Reset detection counter for this frame

    # ===== PROCESS DETECTIONS & UPDATE HEATMAP =====
    for result in results:
        boxes = result.boxes
        
        for box in boxes:
            count += 1

            # Extract bounding box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # Draw green rectangle around detected person
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Calculate center point of detection
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            # Increment heatmap at detection center
            heatmap[cy, cx] += 1

    # ===== COLLECT PER-SECOND DATA FOR GRAPH =====
    current_time = time.time() - start_time
    second_counts.append(count)
    
    # When second boundary is crossed, aggregate and store data point
    if int(current_time) > current_second:
        current_second = int(current_time)
        
        # Calculate minimum count in past second (reduces noise)
        min_count = min(second_counts) if second_counts else 0
        
        # Store data point for graph
        time_data.append(current_second)
        count_data.append(min_count)
        
        # Reset for next second
        second_counts = []

    # Remove old data points to prevent slowdown (keep last 50)
    if len(time_data) > 50:
        time_data = time_data[-50:]
        count_data = count_data[-50:]

    # ===== UPDATE GRAPH EVERY 2 SECONDS =====
    if time.time() - last_graph_update >= 2.0 and len(time_data) > 0:
        last_graph_update = time.time()
        
        # Clear previous graph content
        ax.clear()
        ax.set_facecolor('black')
        
        # Plot data with neon green line
        ax.plot(time_data, count_data, color='#00FF00', linewidth=3)
        
        # Configure axis labels (bold white text)
        ax.set_xlabel("Time (seconds)", fontsize=12, color='#FFFFFF', weight='bold')
        ax.set_ylabel("Persons detected", fontsize=12, color='#FFFFFF', weight='bold')
        
        # Configure grid and tick marks
        ax.grid(True, alpha=0.5, color='#00FF00', linewidth=0.5)
        ax.tick_params(colors='#FFFFFF', labelsize=10)
        
        # Configure axis spines (borders)
        ax.spines['bottom'].set_color('#00FF00')
        ax.spines['top'].set_color('black')
        ax.spines['left'].set_color('#00FF00')
        ax.spines['right'].set_color('black')
        
        # Convert matplotlib figure to OpenCV-compatible image
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        buf = canvas.buffer_rgba()
        size = canvas.get_width_height()
        graph_img = np.frombuffer(buf, dtype=np.uint8).reshape(size[1], size[0], 4)
        graph_img = cv2.cvtColor(graph_img, cv2.COLOR_RGBA2BGR)
        
        # Resize graph to match camera + heatmap combined width
        combined_width = frame_width * 2
        graph_img = cv2.resize(graph_img, (combined_width, graph_img.shape[0]))

    # ===== PROCESS & RENDER HEATMAP =====
    # Apply exponential decay to create fading effect
    heatmap *= 0.95
    
    # Apply Gaussian blur for smooth visualization
    heatmap_blur = cv2.GaussianBlur(heatmap, (51, 51), 0)
    
    # Normalize values to 0-255 range
    heatmap_norm = cv2.normalize(heatmap_blur, None, 0, 255, cv2.NORM_MINMAX)
    heatmap_uint8 = heatmap_norm.astype(np.uint8)
    
    # Apply thermal colormap: black (none) → red (medium) → yellow (high)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_HOT)
    heatmap_color = cv2.resize(heatmap_color, (frame_width, frame_height))

    # ===== COMBINE CAMERA & HEATMAP SIDE-BY-SIDE =====
    combined = np.hstack((frame, heatmap_color))

    # Display current person count (red text in top-left)
    cv2.putText(combined, f"Person Count: {count}",
                (20, 40),
                cv2.FONT_HERSHEY_COMPLEX,
                1,
                (0, 0, 255),
                3)

    # Draw white vertical divider line between camera and heatmap
    mid_x = frame_width
    cv2.line(combined, (mid_x, 0), (mid_x, combined.shape[0]), (255, 255, 255), 2)

    # ===== COMBINE WITH GRAPH =====
    if graph_img is not None:
        # Stack camera/heatmap on top, graph on bottom
        display = np.vstack((combined, graph_img))
        
        # Draw white horizontal divider line
        cv2.line(display, (0, combined.shape[0]), (display.shape[1], combined.shape[0]), (255, 255, 255), 2)
    else:
        # Show placeholder message until first graph update
        placeholder = np.ones((320, combined.shape[1], 3), dtype=np.uint8) * 255
        cv2.putText(placeholder, "Waiting for graph data...",
                    (combined.shape[1]//2 - 150, 160),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 0),
                    2)
        display = np.vstack((combined, placeholder))
        
        # Draw white horizontal divider line
        cv2.line(display, (0, combined.shape[0]), (display.shape[1], combined.shape[0]), (255, 255, 255), 2)

    # ===== DISPLAY FRAME =====
    cv2.imshow("Crowd Monitoring System", display)

    # ===== CHECK FOR EXIT COMMAND =====
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    # Exit if window is closed manually
    if cv2.getWindowProperty("Crowd Monitoring System", cv2.WND_PROP_VISIBLE) < 1:
        break

# ============================================================================
# CLEANUP AND RESOURCE RELEASE
# ============================================================================
cap.release()
cv2.destroyAllWindows()
plt.close(fig)