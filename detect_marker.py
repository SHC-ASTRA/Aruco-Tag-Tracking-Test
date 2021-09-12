import cv2
from cv2 import aruco


def findArucoMarkers(img, markerSize=4, totalMarkers=50, draw=True):
    """
    Finds the aruco markers in the image.
    :param img: The image to find the markers in.
    :param markerSize: The size of the markers to find.
    :param totalMarkers: The total number of markers to find.
    :param draw: Whether or not to draw the markers on the image.
    :return: A list of the markerids found.
    """
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    key = getattr(aruco, f"DICT_{markerSize}X{markerSize}_{totalMarkers}")
    aruco_dict = aruco.Dictionary_get(key)
    aruco_parameters = aruco.DetectorParameters_create()
    bboxes, ids, _ = aruco.detectMarkers(
        imgGray, aruco_dict, parameters=aruco_parameters
    )
    if draw:
        aruco.drawDetectedMarkers(img, bboxes, ids)
    return ids


def main():
    """
    Main function.
    """
    cap = cv2.VideoCapture(0)

    while True:
        # Capture frame-by-frame
        _, frame = cap.read()
        findArucoMarkers(frame)
        cv2.imshow("frame", frame)
        cv2.waitKey(1)


# Run the main function
if __name__ == "__main__":
    main()
