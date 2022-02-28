import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)
p_time = 0

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=2)
mp_draw = mp.solutions.drawing_utils


while True:
    success, img = cap.read()
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(img_rgb)
    print(result.multi_face_landmarks)
    if result.multi_face_landmarks:
        for face_lm in result.multi_face_landmarks:
            mp_draw.draw_landmarks(img, face_lm)

    c_time = time.time()
    fps = 1 / (c_time - p_time)
    p_time = c_time
    cv2.putText(
        img, f'FPS: {int(fps)}',
        (10, 60), cv2.FONT_HERSHEY_PLAIN,
        2, (255, 0, 255), 2
    )

    cv2.imshow("Image", img)
    cv2.waitKey(1)