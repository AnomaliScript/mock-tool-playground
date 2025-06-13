import cv2
import cv2.aruco as aruco
# Type in the function/method in question and the server will give you its (messy) documentation via the terminal
#help(cv2.aruco.CharucoDetector)
import cv2
import cv2.aruco as aruco

# Create the ChArUco board
aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)
board = aruco.CharucoBoard_create(
    squaresX=7,
    squaresY=5,
    squareLength=40,  # pixels
    markerLength=30,  # pixels
    dictionary=aruco_dict
)

# Draw the board
img = board.draw((1000, 700))

# Save to file
cv2.imwrite("charuco_board_printable.png", img)

