import mediapipe as mp
import math
from fingerTrack import *
import cv2
import pyautogui as pag
import numpy as np


screen_width, screen_height = pag.size()
prev_mouse_pos = (0, 0)

def initialize_mediapipe(): #THIS MAY MEED TO BE DELETED
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    mp_drawing = mp.solutions.drawing_utils
    return mp_hands, hands, mp_drawing

def process_frame(frame, hands):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    return results
        

# def new_functionality(frame, results):
    
    #mouseDown = False
    #frame_height, frame_width, _ = frame.shape
    #midpointCalc()
    
    #if gesture == "Pointing" and mouseDown == False:
            #mouse down
        #    pag.mouseDown()
         #   mouseDown = True
    #if gesture != "Pointing" and mouseDown == True:
     #       #mouse up
      #      pag.mouseUp()
       #     mouseDown = False
            
    #if mouseDown:
        #draw a circle at the midpoint with radius 10
     #   cv2.circle(frame, (int(midpoint_x*frame_width), int(midpoint_y*frame_height)), 10, (0, 255, 0), -1)
        
    #else:
        #erase the circle at the midpoint with radius 10
     #   cv2.circle(frame, (int(midpoint_x*frame_width), int(midpoint_y*frame_height)), 10, (0, 255, 0), 1)
        
    #map the position to the screen resolution
    #x_mapped = np.interp(midpoint_x, [0, 1], [0, screen_width])
    #y_mapped = np.interp(midpoint_y, [0, 1], [0, screen_height])
    
    #pag.moveTo (x_mapped, y_mapped, duration = 0.01)

def get_palm_coordinates(hand_landmarks, mp_hands):
    palm_x = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].x
    palm_y = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y
    return palm_x, palm_y

def convert_coordinates_to_pixels(x, y, frame):
    frame_height, frame_width, _ = frame.shape
    x_px = int(x * frame_width)
    y_px = int(y * frame_height)
    return x_px, y_px

def track_hands(cap):
    mp_hands, hands, mp_drawing = initialize_mediapipe()
    mouseDown = False
    controlUp = False
    global prev_mouse_pos
    scroll_threshold = 0.1  # The amount of accumulated delta needed to trigger a scroll
    prev_midpoint_x, prev_midpoint_y = 0, 0
    prev_midpoint_distance = 0
    scroll_sensitivity = 0.1  # Adjust this value to control the scroll sensitivity
    cumulative_delta = 0
    
    ## MAIN LOOP
    while True:
        
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        frame = cv2.flip(frame, 1)
        
        results = process_frame(frame, hands)
        text_positions = [(50, 50), (50, 100)]  # Positions for displaying gesture text
        coord_positions = [(500, 50), (500, 100)]  # Positions for displaying coordinates
        distance_position = (800, 50)  # Position for displaying distance between hands

        hand_coordinates = []
        fist_detected = False
        gun_detected = 0
        midpoint_x_sum, midpoint_y_sum = 0, 0
        
        frame_height, frame_width, _ = frame.shape
        
        if results.multi_hand_landmarks:
            for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)
                )

                gesture = recognize_gesture(hand_landmarks)
            
                text_position = text_positions[i] if i < len(text_positions) else (50, 50 + i * 50)
                cv2.putText(frame, gesture, text_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            
                palm_x, palm_y = get_palm_coordinates(hand_landmarks, mp_hands)
                palm_x_px, palm_y_px = convert_coordinates_to_pixels(palm_x, palm_y, frame)
                coord_position = coord_positions[i] if i < len(coord_positions) else (500, 50 + i * 50)
                cv2.putText(frame, f"({palm_x_px}, {palm_y_px})", coord_position, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

                hand_coordinates.append((palm_x_px, palm_y_px))
                
                index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]

                #get the midpoint between the thumb and index finger
                midpoint_x = (index_finger_tip.x + thumb_tip.x) /2
                midpoint_y = (index_finger_tip.y + thumb_tip.y) /2

                # Get the distance between the thumb and index finger
                distance = np.sqrt((index_finger_tip.x - thumb_tip.x)**2 + (index_finger_tip.y - thumb_tip.y)**2)

                if i == 0:  # First hand controls the mouse
                    if gesture == "Pointing" and mouseDown == False:
                        #mouse down
                        pag.click()
                        pag.mouseDown()
                        mouseDown = True
                        
                    if gesture != "Pointing" and mouseDown == True:
                        #mouse up
                        pag.mouseUp()
                        mouseDown = False
                
                    if mouseDown:
                        #draw a circle at the midpoint with radius 10
                        cv2.circle(frame, (int(midpoint_x*frame_width), int(midpoint_y * frame_height)), 10, (0, 255,0), -1)

                    else:
                        #draw a circle at the midpoint with radius 10
                        cv2.circle(frame, (int(midpoint_x*frame_width), int(midpoint_y * frame_height)), 10, (0, 255,0), 1)
                
                    # Map the position to the screen resolution with a 10% extension on each side
                    extended_range = 0.1
                    x_mapped = np.interp(midpoint_x, (-extended_range, 1 + extended_range), (-screen_width * extended_range, screen_width * (1 + extended_range)))
                    y_mapped = np.interp(midpoint_y, (-extended_range, 1 + extended_range), (-screen_height * extended_range, screen_height * (1 + extended_range)))

                    # Apply exponential smoothing to the mouse position
                    smooth_factor = 0.6  # Adjust this value between 0 and 1 for desired smoothing
                    smooth_x = prev_mouse_pos[0] + smooth_factor * (x_mapped - prev_mouse_pos[0])
                    smooth_y = prev_mouse_pos[1] + smooth_factor * (y_mapped - prev_mouse_pos[1])

                    # Set the mouse position using the smoothed coordinates
                    pag.moveTo(smooth_x, smooth_y, duration=0.001)

                    # Update the previous mouse position
                    prev_mouse_pos = (smooth_x, smooth_y)
                    
                if gesture == "Fist":
                    fist_detected = True
                
                if gesture == "Gun":
                    gun_detected += 1
                    midpoint_x_sum += midpoint_x
                    midpoint_y_sum += midpoint_y
                    
        if fist_detected and controlUp == False:
            pag.hotkey('ctrl', 'up')
            controlUp = True
        
        if not fist_detected and controlUp == True:
            pag.hotkey('ctrl', 'down')
            controlUp = False
        
        if gun_detected == 2:
            midpoint_x_avg = midpoint_x_sum / gun_detected
            midpoint_y_avg = midpoint_y_sum / gun_detected
            
            # Calculate the midpoint between the two midpoints of each hand
            zoom_midpoint_x = int(midpoint_x_avg * frame.shape[1])
            zoom_midpoint_y = int(midpoint_y_avg * frame.shape[0])
            
            # Draw a circle at the zoom midpoint
            cv2.circle(frame, (zoom_midpoint_x, zoom_midpoint_y), 10, (255, 0, 0), -1)
            
            # Calculate the change in distance
            midpoint_distance = np.sqrt((midpoint_x_avg - prev_midpoint_x)**2 + (midpoint_y_avg - prev_midpoint_y)**2)
            delta_distance = midpoint_distance - prev_midpoint_distance
            
            # Accumulate the delta to scroll smoothly
            cumulative_delta += delta_distance * scroll_sensitivity
            if abs(cumulative_delta) >= scroll_threshold:
                pag.scroll(int(cumulative_delta))
                cumulative_delta = 0  # Reset after scrolling
            
            # Update previous values for the next frame
            prev_midpoint_x, prev_midpoint_y = midpoint_x_avg, midpoint_y_avg
            prev_midpoint_distance = midpoint_distance
                
        if len(hand_coordinates) == 2:
            x_distance = abs(hand_coordinates[0][0] - hand_coordinates[1][0])
            y_distance = abs(hand_coordinates[0][1] - hand_coordinates[1][1])
            cv2.putText(frame, f"X Distance: {x_distance}", (distance_position[0], distance_position[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, f"Y Distance: {y_distance}", (distance_position[0], distance_position[1] + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        
        cv2.imshow('Hand Tracking', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    hands.close()