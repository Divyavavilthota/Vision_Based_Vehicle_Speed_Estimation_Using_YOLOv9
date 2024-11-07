import cv2
import torch
import numpy as np
import argparse
from deep_sort_realtime.deepsort_tracker import DeepSort
import time
import pyttsx3
import threading
from ultralytics import YOLO

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

def main():
    opt = parse_args()  # Define 'opt' here
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
    prev_positions = {}
    speed_accumulator = {}
    
    danger_cars = []  # List to store car numbers in danger
    already_warned = set()  # Set to track already warned cars
    total_time_taken_ns = 0  # Initialize total time taken

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

            if track_id in prev_positions:
                prev_position = prev_positions[track_id]
                distance = calculate_distance(prev_position, transformed_pt[0][0])

                # Measure time taken to calculate average speed
                start_time = time.perf_counter_ns()
                speed = calculate_speed(distance, fps)
                end_time = time.perf_counter_ns()
                time_taken_ns = end_time - start_time
                total_time_taken_ns = time_taken_ns  # Only update once

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

            avg_speed = sum(speed_accumulator[track_id]) / len(speed_accumulator[track_id]) if track_id in speed_accumulator else 0
            status_text = "Safe" if avg_speed <= 150 else "Danger"

            # Draw the bounding box
            if status_text == "Danger":
                if track_id not in already_warned:
                    threading.Thread(target=speak_text, args=(f"Car {track_id} is in danger!",)).start()
                    already_warned.add(track_id)  # Mark as warned
                box_color = (0, 0, 255)  # Red for danger
            else:
                if track_id in already_warned:
                    already_warned.remove(track_id)  # Reset warning status
                box_color = (0, 255, 0)  # Green for safe

            frame = draw_corner_rect(frame, (x1, y1, x2 - x1, y2 - y1), rect_color=box_color)

            speed_text = f"Speed: {avg_speed:.1f} km/h"

            # Draw speed text above the car
            cv2.rectangle(frame, (x1 - 1, y1 - 20), (x1 + len(speed_text) * 10, y1), box_color, -1)
            cv2.putText(frame, speed_text, (x1 + 5, y1 - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            # Draw status text above the speed text
            cv2.rectangle(frame, (x1 - 1, y1 - 40), (x1 + len(status_text) * 10, y1 - 20), box_color, -1)
            cv2.putText(frame, status_text, (x1 + 5, y1 - 27), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            # Update the danger cars list
            if status_text == "Danger":
                danger_cars.append(track_id)
            else:
                if track_id in danger_cars:
                    danger_cars.remove(track_id)

        # Display total time taken in the corner
        time_text = f"Time taken: {total_time_taken_ns} ns"
        cv2.putText(frame, time_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Write the output frame
        writer.write(frame)

        # Display the frame
        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
