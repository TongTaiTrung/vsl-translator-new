"""Draw pose and hand landmarks on BGR frames."""

from src.builtin.mediapipe_builtin import defs
import cv2
import numpy as np


def draw(frame, results, point_color=(0,0,0), line_color=(255,255,255), secondary_color=(0,0,0)):
    if not results: return frame
    connections = defs.HandConnections

    if hasattr(results, 'hand_landmarks'): multi_landmarks = results.hand_landmarks
    elif hasattr(results, 'pose_landmarks'): 
        multi_landmarks = results.pose_landmarks
        connections = defs.PoseConnections

    else: return frame

    if not multi_landmarks: return frame
    height, width, _ = frame.shape

    is_hand = hasattr(results, 'hand_landmarks')
    handedness = getattr(results, 'handedness', []) if is_hand else []
    for index, landmarks in enumerate(multi_landmarks):
        current_color = point_color
        current_line_color = line_color
        if index < len(handedness) and handedness[index]:
            if handedness[index][0].category_name == 'Left':
                current_color = secondary_color
                current_line_color = secondary_color
        
        if connections:
            for connection in connections:
                start_idx, end_idx = connection[0], connection[1]
                if start_idx < len(landmarks) and end_idx < len(landmarks):
                    start_pt = landmarks[start_idx]
                    end_pt = landmarks[end_idx]
                
                    start_xy = (int(start_pt.x * width), int(start_pt.y * height))
                    end_xy = (int(end_pt.x * width), int(end_pt.y * height))
                    
                    cv2.line(frame, start_xy, end_xy, current_line_color, 2)

        for point in landmarks:
            center_x, center_y = int(point.x * width), int(point.y * height)
            cv2.circle(frame, (center_x, center_y), 4, current_color, -1)

    return frame