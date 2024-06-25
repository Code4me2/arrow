import mediapipe as mp
import math
from fist import *
import numpy as np

gesture_threshold = 0.05
fist_threshold = 0.2
fingers_together_threshold = 0.03
thumb_distance_threshold = 0.25

def calculate_finger_distances(thumb_tip, index_tip, middle_tip, ring_tip, pinky_tip):
    # Calculate the distances between the thumb tip and other finger tips
    thumb_index_distance = math.hypot(thumb_tip.x - index_tip.x, thumb_tip.y - index_tip.y)
    thumb_middle_distance = math.hypot(thumb_tip.x - middle_tip.x, thumb_tip.y - middle_tip.y)
    thumb_ring_distance = math.hypot(thumb_tip.x - ring_tip.x, thumb_tip.y - ring_tip.y)
    thumb_pinky_distance = math.hypot(thumb_tip.x - pinky_tip.x, thumb_tip.y - pinky_tip.y)
    return thumb_index_distance, thumb_middle_distance, thumb_ring_distance, thumb_pinky_distance

def are_fingers_together(hand_landmarks, thumb_finger_distances):
    index_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP]
    index_dip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_DIP]
    index_pip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_PIP]
    
    middle_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.MIDDLE_FINGER_TIP]
    middle_dip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.MIDDLE_FINGER_DIP]
    middle_pip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.MIDDLE_FINGER_PIP]
    
    tip_distance = abs(index_tip.x - middle_tip.x)
    dip_distance = abs(index_dip.x - middle_dip.x)
    pip_distance = abs(index_pip.x - middle_pip.x)
    
    thumb_index_distance, thumb_middle_distance, _, _ = thumb_finger_distances
    
    if tip_distance < fingers_together_threshold and dip_distance < fingers_together_threshold and pip_distance < fingers_together_threshold and thumb_index_distance > thumb_distance_threshold and thumb_middle_distance > thumb_distance_threshold:
        return True
    return False
    
def determine_gesture(thumb_finger_distances, finger_knuckle_distances, thumb_tip, index_second_joint, middle_third_joint, hand_landmarks):
    thumb_index_distance, thumb_middle_distance, thumb_ring_distance, thumb_pinky_distance = thumb_finger_distances
    
    if thumb_index_distance < gesture_threshold and all(distance > gesture_threshold for distance in thumb_finger_distances[1:]):
        return "Pointing"
    elif are_fingers_together(hand_landmarks, thumb_finger_distances):
        return "Gun"
    elif all(distance > gesture_threshold for distance in thumb_finger_distances):
        return "Open Hand"
    elif is_fist(finger_knuckle_distances, fist_threshold) and is_thumb_close_to_joint(thumb_tip, index_second_joint, middle_third_joint, fist_threshold):
        return "Fist"
    else:
        return "Unknown"

def recognize_gesture(hand_landmarks):
    # Define the landmark indices for each finger
    thumb_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.THUMB_TIP]
    index_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.RING_FINGER_TIP]
    pinky_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.PINKY_TIP]
    
    # Calculate the distances between the thumb tip and other finger tips
    thumb_finger_distances = calculate_finger_distances(thumb_tip, index_tip, middle_tip, ring_tip, pinky_tip)
    
    # Calculate the distances between each finger tip and corresponding knuckle, and get the second joint of the index finger and the third joint of the middle finger
    finger_knuckle_distances, index_second_joint, middle_third_joint = calculate_knuckle_distances(hand_landmarks)
    
    # Recognize gestures based on the distances
    gesture = determine_gesture(thumb_finger_distances, finger_knuckle_distances, thumb_tip, index_second_joint, middle_third_joint, hand_landmarks)
    
    return gesture