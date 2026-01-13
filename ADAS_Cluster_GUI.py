import tkinter as tk
from PIL import Image, ImageTk
import cv2
from ADAS_Lane_Detection import pipeline

# Load the video
cap = cv2.VideoCapture("test_video.mp4")
if not cap.isOpened():
    print("Error: Could not open video file", cap)
    exit(1)


# Create the main window
root = tk.Tk()
root.title("ADAS Cluster GUI")
root.geometry("800x600")

# Video display label
video_label = tk.Label(root)
video_label.pack()

# Status label
status_label = tk.Label(root, text="Initializing...", font=("Arial", 24))
status_label.pack()

def update_frame():
    ret, frame = cap.read()
    frame = cv2.resize(frame, (800, 450))
    if ret:
        # Process the frame
        processed, lane_center = pipeline(frame)
        
        # Convert to PIL Image
        img = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        imgtk = ImageTk.PhotoImage(image=img)
        
        # Update the label
        video_label.imgtk = imgtk
        video_label.configure(image=imgtk)
        
        # Determine status
        if lane_center is not None:
            height, width = frame.shape[:2]
            image_center = width / 2
            on_track = abs(lane_center - image_center) < 50  # Threshold for on track
            status_text = "On Track" if on_track else "Off Track"
            color = "green" if on_track else "red"
        else:
            status_text = "Lane Not Detected"
            color = "orange"
        
        status_label.config(text=status_text, fg=color)
        
        # Schedule next update
        root.after(30, update_frame)  # ~30 FPS
    else:
        # Video ended, restart or stop
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        root.after(30, update_frame)

# Start updating frames
update_frame()

# Run the GUI
root.mainloop()

# Release the video capture
cap.release()