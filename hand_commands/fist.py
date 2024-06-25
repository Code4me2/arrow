import mediapipe as mp
import math

def calculate_knuckle_distances(hand_landmarks):
    # Define the landmark indices for each finger tip, corresponding knuckle, and additional joints
    finger_joints = [
        (mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP, mp.solutions.hands.HandLandmark.INDEX_FINGER_MCP, mp.solutions.hands.HandLandmark.INDEX_FINGER_PIP),
        (mp.solutions.hands.HandLandmark.MIDDLE_FINGER_TIP, mp.solutions.hands.HandLandmark.MIDDLE_FINGER_MCP, mp.solutions.hands.HandLandmark.MIDDLE_FINGER_DIP),
        (mp.solutions.hands.HandLandmark.RING_FINGER_TIP, mp.solutions.hands.HandLandmark.RING_FINGER_MCP, None),
        (mp.solutions.hands.HandLandmark.PINKY_TIP, mp.solutions.hands.HandLandmark.PINKY_MCP, None)
    ]
    
    distances = []
    index_second_joint = None
    middle_third_joint = None
    
    for finger_tip, knuckle, joint in finger_joints:
        tip = hand_landmarks.landmark[finger_tip]
        knuckle = hand_landmarks.landmark[knuckle]
        distance = math.hypot(tip.x - knuckle.x, tip.y - knuckle.y)
        distances.append(distance)
        
        if joint is not None:
            if joint == mp.solutions.hands.HandLandmark.INDEX_FINGER_PIP:
                index_second_joint = hand_landmarks.landmark[joint]
            elif joint == mp.solutions.hands.HandLandmark.MIDDLE_FINGER_DIP:
                middle_third_joint = hand_landmarks.landmark[joint]
    
    return distances, index_second_joint, middle_third_joint

def is_thumb_close_to_joint(thumb_tip, index_second_joint, middle_third_joint, threshold):
    if index_second_joint is not None:
        distance_to_index_second_joint = math.hypot(thumb_tip.x - index_second_joint.x, thumb_tip.y - index_second_joint.y)
        if distance_to_index_second_joint < threshold:
            return True
    
    if middle_third_joint is not None:
        distance_to_middle_third_joint = math.hypot(thumb_tip.x - middle_third_joint.x, thumb_tip.y - middle_third_joint.y)
        if distance_to_middle_third_joint < threshold:
            return True
    
    return False

def is_fist(distances, threshold):
    return all(distance < threshold for distance in distances)
