import mediapipe as mp
import cv2
import pytesseract
from PIL import Image
import numpy as np
import uuid
import os
import math 
pytesseract.pytesseract.tesseract_cmd = "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
cap = cv2.VideoCapture(0)
pages = []
far_point = []
temp_far_points = []
def dist(point1, point2):
    return math.sqrt((point1[0] - point2[0])*(point1[0] - point2[0]) + (point1[1] - point2[1])*(point1[1] - point2[1]) )

with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5, max_num_hands=1) as hands:
    while cap.isOpened():
        ret, frame = cap.read()

        # BGR 2 RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Flip on horizontal
        image = cv2.flip(image, 1)

        # Set flag
        image.flags.writeable = False

        # Detections
        results = hands.process(image)

        # Set flag to true
        image.flags.writeable = True

        # RGB 2 BGR
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Detections
        image_height, image_width, _ = image.shape

        # ocr_frame = cv2.imread("filename.jpeg")

        for j in range(len(temp_far_points) - 1):
            cv2.line(image, temp_far_points[j],
                        temp_far_points[j+1], (255, 5, 255), 5)

        for i in range(len(far_point)):
            if len(far_point[i]) > 1:
                for j in range(len(far_point[i]) - 1):
                        cv2.line(image, far_point[i][j],
                                far_point[i][j+1], (255, 5, 255), 5)
                        # cv2.line(ocr_frame, far_point[i][j],
                        #         far_point[i][j+1], (255, 255, 255), 5)
        # Rendering results
        if results.multi_hand_landmarks:
            for num, hand in enumerate(results.multi_hand_landmarks):
                for hand_landmarks in results.multi_hand_landmarks:
                    #                     print('hand_landmarks:', hand_landmarks)
                    distance = dist((int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].x * image_width), int(
                        hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y * image_height)), (int(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x * image_width), int(
                        hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y * image_height)))
                    distance_erase_all = dist((int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width), int(
                        hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height)), (int(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x * image_width), int(
                        hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y * image_height)))

                    distance_go_back = dist((int(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].x * image_width), int(
                        hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y * image_height)), (int(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x * image_width), int(
                        hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y * image_height)))
                    
                    if(distance_erase_all < 25):
                        if len(far_point) != 0:
                            pages.append(far_point)
                            far_point = []
                            temp_far_points = []
            
                    if(distance_go_back < 25):
                        far_point.append(pages[-1][0])

                    if distance < 25:
                        temp_far_points.append((int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width), int(
                            hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height)))
                    else:
                        if len(temp_far_points) != 0:
                            far_point.append(temp_far_points)
                            temp_far_points = []
                        # text = pytesseract.image_to_string(ocr_frame, config=("-c tessedit"
                        #                                     "_char_whitelist=abcdefghijklmnopqrstuvwxyz0123456789"
                        #                                     " --psm 10"
                        #                                     " -l osd"
                        #                                     " "))

                        # print(text)

                cv2.circle(image, (int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width), int(
                            hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height)) , 8, (0, 0, 255), -1)
                cv2.circle(image, (int(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x * image_width), int(
                            hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y * image_height)) , 8, (0, 255, 0), -1)
                cv2.circle(image, (int(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].x * image_width), int(
                            hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].y * image_height)) , 8, (255, 0, 0), -1)
                cv2.circle(image, (int(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x * image_width), int(
                            hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y * image_height)) , 8, (255, 255, 0), -1)

                # for i in range(len(far_point)):
                #     if len(far_point[i]) > 1:
                #         for j in range(len(far_point[i]) - 1):
                #                 cv2.line(ocr_frame, far_point[i][j],
                #                         far_point[i][j+1], (255, 255, 255), 5)

        cv2.imshow('Hand Tracking', image)
        # cv2.imshow('OCR', ocr_frame )
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
