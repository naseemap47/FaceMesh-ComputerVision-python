import cv2
import mediapipe as mp
import time

def face_mesh(
    num_faces=2,
    p_time=0, draw_thick=1, draw_circle_radius=1,
    draw_color=(0, 255, 0), display_face_id=False,
    landmark_connections=True, face_id_font_scale=1,
    face_id_color=(0, 255, 0), face_id_thickness=1
):
    cap = cv2.VideoCapture(0)

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(max_num_faces=num_faces)
    mp_draw = mp.solutions.drawing_utils
    draw_spec = mp_draw.DrawingSpec(
        thickness=draw_thick,
        circle_radius=draw_circle_radius,
        color = draw_color
    )

    while True:
        success, img = cap.read()
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = face_mesh.process(img_rgb)
        
        if result.multi_face_landmarks:
            for face_lm in result.multi_face_landmarks:

                if landmark_connections:
                    connections=mp_face_mesh.FACEMESH_CONTOURS
                else:
                    connections=None
            
                mp_draw.draw_landmarks(
                    image=img, landmark_list=face_lm,
                    connections=connections,
                    landmark_drawing_spec=draw_spec
                )
                if display_face_id:
                    for id, lm in enumerate(face_lm.landmark):
                        # print(id, lm)
                        height, width, channel = img.shape
                        x, y = int(lm.x * width), int(lm.y * height)
                        cv2.putText(
                            img, str(id), (x, y),
                            cv2.FONT_HERSHEY_PLAIN,
                            face_id_font_scale, face_id_color,
                            face_id_thickness
                        )

            

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
