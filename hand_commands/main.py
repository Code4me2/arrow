import cv2
from fingerTrack import *
from handTracker import *

def access_camera(camera_index=0):
    # Open the default camera (0) or a specified camera
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise ValueError("Cannot open camera")
    return cap
    
def test_camera():
    cap = access_camera()  # Change the index if using a different camera
    track_hands(cap)
    
if __name__ == "__main__":
    try:
        test_camera()
    except Exception as e:
        print(f"An error occurred: {e}")
