import cv2
from pupil_apriltags import Detector
import numpy as np

# Initialize webcam
cap = cv2.VideoCapture(0)

# Initialize AprilTag detector (default uses tag36h11 family)
at_detector = Detector(
    families='tag36h11',
    nthreads=1,
    quad_decimate=0.75,
    quad_sigma=0.0,
    refine_edges=True,
    decode_sharpening=0.75,
    debug=False
)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect tags
    results = at_detector.detect(gray)

    # Drawing
    for r in results:
        (ptA, ptB, ptC, ptD) = r.corners
        ptA = tuple(map(int, ptA))
        ptB = tuple(map(int, ptB))
        ptC = tuple(map(int, ptC))
        ptD = tuple(map(int, ptD))

        # Draw bounding box
        cv2.line(frame, ptA, ptB, (0, 255, 0), 2)
        cv2.line(frame, ptB, ptC, (0, 255, 0), 2)
        cv2.line(frame, ptC, ptD, (0, 255, 0), 2)
        cv2.line(frame, ptD, ptA, (0, 255, 0), 2)

        # Draw tag center
        (cX, cY) = tuple(map(int, r.center))
        cv2.circle(frame, (cX, cY), 5, (0, 0, 255), -1)

        # Draw tag ID
        tag_id = r.tag_id
        cv2.putText(frame, str(tag_id), (ptA[0], ptA[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        print(f"Detected ID: {r.tag_id}, Family: {r.tag_family}")

    # Show result
    cv2.imshow("AprilTag Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
