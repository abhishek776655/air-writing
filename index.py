import mediapipe as mp
import cv2
import numpy as np
import uuid
import os
import math 

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(0)
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
        print(results)
        image_height, image_width, _ = image.shape

        # Rendering results
        if results.multi_hand_landmarks:
            for num, hand in enumerate(results.multi_hand_landmarks):
                for hand_landmarks in results.multi_hand_landmarks:
                    #                     print('hand_landmarks:', hand_landmarks)
                    distance = dist((int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].x * image_width), int(
                        hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y * image_height)), (int(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x * image_width), int(
                        hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y * image_height)))
                    print(far_point)
                    distance_erase_all = dist((int(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x * image_width), int(
                        hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y * image_height)), (int(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x * image_width), int(
                        hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y * image_height)))
                    
                    if(distance_erase_all < 25):
                        far_point.clear()
                        temp_far_points.clear()
                    if distance < 25:
                        temp_far_points.append((int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width), int(
                            hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height)))
                    else:
                        if len(temp_far_points) != 0:
                            far_point.append(temp_far_points)
                            temp_far_points = []

                    for j in range(len(temp_far_points) - 1):
                        cv2.line(image, temp_far_points[j],
                                 temp_far_points[j+1], (255, 5, 255), 10)

                    for i in range(len(far_point)):
                        for j in range(len(far_point[i]) - 1):
                            cv2.line(image, far_point[i][j],
                                 far_point[i][j+1], (255, 5, 255), 10)


#                     print(
#                         f'Index finger tip coordinates: (',
#                         f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width}, '
#                         f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height})'
#                     )
                mp_drawing.draw_landmarks(image, hand, mp_hands.HAND_CONNECTIONS,
                                          mp_drawing.DrawingSpec(
                                              color=(121, 22, 76), thickness=2, circle_radius=4),
                                          mp_drawing.DrawingSpec(
                                              color=(250, 44, 250), thickness=2, circle_radius=2),
                                          )

        cv2.imshow('Hand Tracking', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
