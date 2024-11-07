import cv2
import torch
import numpy as np
import argparse
from deep_sort_realtime.deepsort_tracker import DeepSort
import time
from ultralytics import YOLO
import pyttsx3
import threading
import pandas as pd
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--video",
        type=str,
        nargs="?",
        default="content/highway_mini.mp4",
        help="Path to input video"
    )
    parser.add_argument(
        "--output",
        type=str,
        nargs="?",
        help="path to output video",
        default="content/output1.mp4"
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.50,
        help="confidence threshold",
    )
    parser.add_argument(
        "--blur_id",
        type=int,
        default=None,
        help="class ID to apply Gaussian Blur",
    )
    parser.add_argument(
        "--class_id",
        type=int,
        default=None,
        help="class ID to track",
    )
    opt = parser.parse_args()
    return opt

def draw_corner_rect(img, bbox, line_length=30, line_thickness=5, rect_thickness=1,
                     rect_color=(255, 0, 255), line_color=(0, 255, 0)):
    x, y, w, h = bbox
    x1, y1 = x + w, y + h

    if rect_thickness != 0:
        cv2.rectangle(img, bbox, rect_color, rect_thickness)

    # Top Left
    cv2.line(img, (x, y), (x + line_length, y), line_color, line_thickness)
    cv2.line(img, (x, y), (x, y + line_length), line_color, line_thickness)

    # Top Right
    cv2.line(img, (x1, y), (x1 - line_length, y), line_color, line_thickness)
    cv2.line(img, (x1, y), (x1, y + line_length), line_color, line_thickness)

    # Bottom Left
    cv2.line(img, (x, y1), (x + line_length, y1), line_color, line_thickness)
    cv2.line(img, (x, y1), (x, y1 - line_length), line_color, line_thickness)

    # Bottom Right
    cv2.line(img, (x1, y1), (x1 - line_length, y1), line_color, line_thickness)
    cv2.line(img, (x1, y1), (x1, y1 - line_length), line_color, line_thickness)

    return img  

def calculate_speed(distance, fps):
    return (distance * fps) * 3.6

def calculate_distance(p1, p2):
    return np.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)

def read_frames(cap):
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        yield frame 

def speak_text(text):
    engine = pyttsx3.init()  # Initialize TTS engine
    engine.say(text)
    engine.runAndWait()

def main(_argv):
    FRAME_WIDTH = 30
    FRAME_HEIGHT = 100

    SOURCE_POLYGONE = np.array([[18, 550], [1852, 608], [1335, 370], [534, 343]], dtype=np.float32)
    BIRD_EYE_VIEW = np.array([[0, 0], [FRAME_WIDTH, 0], [FRAME_WIDTH, FRAME_HEIGHT], [0, FRAME_HEIGHT]], dtype=np.float32)

    M = cv2.getPerspectiveTransform(SOURCE_POLYGONE, BIRD_EYE_VIEW)

    # Initialize the video capture
    video_input = opt.video
    cap = cv2.VideoCapture(video_input)
    if not cap.isOpened():
        print('Error: Unable to open video source.')
        return

    frame_generator = read_frames(cap)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    pts = SOURCE_POLYGONE.astype(np.int32)
    pts = pts.reshape((-1, 1, 2))

    polygon_mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
    cv2.fillPoly(polygon_mask, [pts], 255)

    # Video writer objects
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(opt.output, fourcc, fps, (frame_width, frame_height))

    # Initialize the DeepSort tracker
    tracker = DeepSort(max_age=50)
    model = YOLO("yolov10n.pt")
    classes_path = "configs/coco.names"
    with open(classes_path, "r") as f:
        class_names = f.read().strip().split("\n")

    np.random.seed(42)
    colors = np.random.randint(0, 255, size=(len(class_names), 3))
    frame_count = 0
    start_time = time.time()
    prev_positions = {}
    speed_accumulator = {}
    
    danger_cars = []  # List to store car numbers in danger

    while True:
        try:
            frame = next(frame_generator)
        except StopIteration:
            break

        # Run model on each frame
        with torch.no_grad():
            results = model(frame)

        detect = []
        for pred in results:
            for box in pred.boxes:    
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = box.conf[0]     
                label = box.cls[0]  

                # Filter out weak detections
                if opt.class_id is None:
                    if confidence < opt.conf:
                        continue
                else:
                    if class_id != opt.class_id or confidence < opt.conf:
                        continue            

                if polygon_mask[(y1 + y2) // 2, (x1 + x2) // 2] == 255:
                    detect.append([[x1, y1, x2 - x1, y2 - y1], confidence, int(label)])            

        tracks = tracker.update_tracks(detect, frame=frame)
        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id    
            ltrb = track.to_ltrb()
            class_id = track.get_det_class()
            x1, y1, x2, y2 = map(int, ltrb)

            if polygon_mask[(y1 + y2) // 2, (x1 + x2) // 2] == 0:
                tracks.remove(track)

            color = colors[class_id]
            B, G, R = map(int, color)
            text = f"{track_id} - {class_names[class_id]}"
            center_pt = np.array([[(x1 + x2) // 2, (y1 + y2) // 2]], dtype=np.float32)
            transformed_pt = cv2.perspectiveTransform(center_pt[None, :, :], M)
            time_log = []


            if track_id in prev_positions:
                prev_position = prev_positions[track_id]
                distance = calculate_distance(prev_position, transformed_pt[0][0])
                start_time = time.perf_counter_ns()
                speed = calculate_speed(distance, fps)
                end_time = time.perf_counter_ns()
                time_taken_ns = end_time - start_time
                total_time_taken_ns = time_taken_ns
                time_text = f"Time taken: {total_time_taken_ns} ns"
                time_log.append((track_id, total_time_taken_ns))

# At the end of the main function, save the log
                pd.DataFrame(time_log, columns=['Track_ID', 'Time_Taken']).to_csv('yolov9_time_log.csv', index=False)

    # Set the color to blue for text
                text_color = (255, 0, 0)  # Blue color in BGR format
                background_color = (0, 0, 0)  # Black background for contrast

    # Calculate text size for positioning
                (text_width, text_height), baseline = cv2.getTextSize(time_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 3)

    # Set positions to ensure they are within frame
                x_position = 10  # Fixed left padding
                y_position = 50  # Set y-position at 50 pixels from the top

    # Draw a filled rectangle behind the text
                cv2.rectangle(frame, (x_position, y_position - text_height - baseline - 5), 
                  (x_position + text_width, y_position + baseline), background_color, -1)
    
    # Put text on the frame with larger font size and thickness
                cv2.putText(frame, time_text, (x_position, y_position), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 3)  # Larger font and thicker text

                writer.write(frame)
                if track_id in speed_accumulator:
                    speed_accumulator[track_id].append(speed)
                    if len(speed_accumulator[track_id]) > 100:
                        speed_accumulator[track_id].pop(0)
                else:
                    speed_accumulator[track_id] = []
                    speed_accumulator[track_id].append(speed)

            prev_positions[track_id] = transformed_pt[0][0]

            # Draw bounding box and text
            box_color = (B, G, R)  # Default color
            line_thickness = 3  # Default thickness
            if track_id in danger_cars:
                box_color = (0, 0, 255)  # Red color for danger
                line_thickness = 8  # Thicker line for danger

            frame = draw_corner_rect(frame, (x1, y1, x2 - x1, y2 - y1), line_length=15, line_thickness=line_thickness, rect_thickness=1, rect_color=box_color, line_color=(255, 255, 255))
            cv2.rectangle(frame, (x1 - 1, y1 - 20), (x1 + len(text) * 10, y1), box_color, -1)
            cv2.putText(frame, text, (x1 + 5, y1 - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            if track_id in speed_accumulator:
                avg_speed = sum(speed_accumulator[track_id]) / len(speed_accumulator[track_id])
                cv2.rectangle(frame, (x1 - 1, y1 - 40), (x1 + len(f"Speed: {avg_speed:.0f} km/h") * 10, y1 - 20), (0, 0, 255), -1)
                cv2.putText(frame, f"Speed: {avg_speed:.0f} km/h", (x1, y1 - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                # Check speed for danger or safe
                if avg_speed <= 150:
                    cv2.rectangle(frame, (x1 - 1, y1 - 60), (x1 + len("SAFE") * 10, y1 - 40), (0, 255, 0), -1)  # Green color
                    cv2.putText(frame, "SAFE", (x1, y1 - 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                elif avg_speed > 200:
                    cv2.rectangle(frame, (x1 - 1, y1 - 60), (x1 + len("DANGER") * 10, y1 - 40), (0, 0, 255), -1)  # Red color
                    cv2.putText(frame, "DANGER", (x1, y1 - 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    
                    # Announce danger in a separate thread
                    threading.Thread(target=speak_text, args=(f"Car number {track_id} is in danger",)).start()
                    
                    # Add track_id to danger cars list
                    if track_id not in danger_cars:
                        danger_cars.append(track_id)

            # Apply Gaussian Blur
            if opt.blur_id is not None and class_id == opt.blur_id:
                if 0 <= x1 < x2 <= frame.shape[1] and 0 <= y1 < y2 <= frame.shape[0]:
                    frame[y1:y2, x1:x2] = cv2.GaussianBlur(frame[y1:y2, x1:x2], (99, 99), 3)

        cv2.polylines(frame, [pts], isClosed=True, color=(255, 0, 0), thickness=2)
        cv2.putText(frame, f"Height: {FRAME_HEIGHT}", (1500, 900), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.putText(frame, f"Width: {FRAME_WIDTH}", (1530, 930), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.imshow('speed_estimation-yolov9', frame)
        writer.write(frame)
        frame_count += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Write danger car numbers to a file
    with open("danger_cars.txt", "w") as f:
        for car_number in danger_cars:
            f.write(f"{car_number}\n")

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    opt = parse_args()
    main(opt)
