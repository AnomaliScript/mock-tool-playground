import cv2
import os

pattern_size = (9, 6)  # Inner corners (columns - 1, rows - 1)
save_folder = "calibration_images"

cap = cv2.VideoCapture(0)
img_id = 0
session_name = input("Which session is this? ")

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCorners(gray, pattern_size)
    display = frame.copy()

    if found:
        cv2.drawChessboardCorners(display, pattern_size, corners, found)

    cv2.imshow('Calibration Capture', display)
    key = cv2.waitKey(1)
    if key == ord('s') and found:
        cv2.imwrite(f"{save_folder}/{session_name}img_{img_id}.png", frame)
        print(f"Saved img_{img_id}.png")
        img_id += 1
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()