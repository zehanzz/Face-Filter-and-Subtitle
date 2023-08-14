# DSP Lab project Face Mask
import cv2
import dlib
from math import hypot

# Create the face detector and the predictor to mark the face points
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Define the flag for different mask
Mask = False
White = False
Opera = False
# Open the webcam
cap = cv2.VideoCapture(0)

# Read the mask file
mask_image = cv2.imread("V.jpeg")
white_image = cv2.imread("oie_oie_trim_image.png")
opera_image = cv2.imread("opera.png")

while True:
    # Create the frame
    _, frame = cap.read()
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Create the first mask
    if Mask:
        faces = detector(grey)
        for face in faces:
            face_mark = predictor(grey, face)
            # Find the position for each part
            center = (face_mark.part(30).x, face_mark.part(30).y)
            left = (face_mark.part(2).x, face_mark.part(2).y)
            right = (face_mark.part(14).x, face_mark.part(14).y)

            # Calculate the width and height of the face
            width = int(hypot(left[0] - right[0], left[1] - right[1])*1)
            height = int(width * 1.3)

            # Mask position
            node1 = (int(center[0] - width / 2), int(center[1] - height / 2))
            node2 = (int(center[0] + width / 2), int(center[1] + height / 2))

            # Resize the mask image to the size of face
            mask = cv2.resize(mask_image, (width, height))

            # Create the mask frame
            mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            _, p1_mask = cv2.threshold(mask_gray, 25, 255, cv2.THRESH_BINARY_INV)
            p1_area = frame[node1[1]: node1[1] + height, node1[0]: node1[0] + width]
            area = cv2.bitwise_and(p1_area, p1_area, mask=p1_mask)

            # Add the mask to the original frame
            final_mask = cv2.add(area, mask)
            frame[node1[1]: node1[1] + height, node1[0]: node1[0] + width] = final_mask

    # Create the first mask
    if White:
        faces = detector(frame)
        for face in faces:
            face_mark = predictor(grey, face)

            # Find the position for each part
            center = (face_mark.part(27).x, face_mark.part(27).y)
            left = (face_mark.part(36).x, face_mark.part(36).y)
            right = (face_mark.part(45).x, face_mark.part(45).y)

            # Calculate the width and height of the face
            width = int(hypot(left[0] - right[0], left[1] - right[1])*1.3)
            height = int(width * 0.44)
            width = int(width * 1.5)

            # Mask position
            node1 = (int(center[0] - width / 2), int(center[1] - height / 2))
            node2 = (int(center[0] + width / 2), int(center[1] + height / 2))

            # Resize the mask image to the size of face
            mask = cv2.resize(white_image, (width, height))

            # Create the mask frame
            mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            # _, p1_mask = cv2.threshold(mask_gray, 25, 255, cv2.THRESH_BINARY_INV)
            _, p1_mask = cv2.threshold(mask_gray, 0, 255, 3)
            p1_area = frame[node1[1]: node1[1] + height, node1[0]: node1[0] + width]
            area = cv2.bitwise_and(p1_area, p1_area, mask=p1_mask)

            # Add the mask to the original frame
            final_mask = cv2.add(area, mask)
            frame[node1[1]: node1[1] + height, node1[0]: node1[0] + width] = final_mask
    if Opera:
        faces = detector(frame)
        for face in faces:
            face_mark = predictor(grey, face)

            # Find the position for each part
            center = (face_mark.part(27).x, face_mark.part(27).y)
            left = (face_mark.part(2).x, face_mark.part(2).y)
            right = (face_mark.part(14).x, face_mark.part(14).y)

            # Calculate the width and height of the face
            width = int(hypot(left[0] - right[0], left[1] - right[1]))
            height = int(width * 1.3)

            # Mask position
            node1 = (int(center[0] - width / 2), int(center[1] - height / 2))
            # node2 = (int(center[0] + width / 2), int(center[1] + height / 2))
            node2 = (int(face_mark.part(14).x), int(face_mark.part(8).y))
            # Resize the mask image to the size of face
            mask = cv2.resize(opera_image, (width, height))

            # Create the mask frame
            mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            _, p1_mask = cv2.threshold(mask_gray, 25, 255, cv2.THRESH_BINARY_INV)
            p1_area = frame[node1[1]: node1[1] + height, node1[0]: node1[0] + width]
            area = cv2.bitwise_and(p1_area, p1_area, mask=p1_mask)

            # Add the mask to the original frame
            final_mask = cv2.add(area, mask)
            frame[node1[1]: node1[1] + height, node1[0]: node1[0] + width] = final_mask
    # Show the frame
    cv2.imshow("Real-time mask", frame)

    # Enable the key control
    key = cv2.waitKey(1)

    if key == ord('s'):
        break
    elif key == ord('m'):
        White = False
        Opera = False
        Mask = ~Mask
    elif key == ord('g'):
        Mask = False
        Opera = False
        White = ~White
    elif key == ord('h'):
        Mask = False
        White = False
        Opera = ~Opera



