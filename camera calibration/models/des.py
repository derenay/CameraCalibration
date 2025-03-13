import cv2
import mediapipe as mp
import math

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)

def calculate_angle(point1, point2, frame_shape):
    """ İki nokta arasındaki açıyı ve piksel koordinatlarını hesaplar """
    image_height, image_width, _ = frame_shape
    

    x1 = int(point1.x * image_width)
    y1 = int(point1.y * image_height)
    x2 = int(point2.x * image_width)
    y2 = int(point2.y * image_height)
    
    dx = x2 - x1
    dy = y2 - y1
    angle = math.degrees(math.atan2(dy, dx))
    return angle, (x1, y1), (x2, y2)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            
            right_eye = face_landmarks.landmark[263]
            left_eye = face_landmarks.landmark[33]
            
            # Açı ve koordinatları hesapla
            roll_angle, left_point, right_point = calculate_angle(left_eye, right_eye, frame.shape)
            
            # Gözler arası çizgiyi çiz
            cv2.line(frame, left_point, right_point, (0, 255, 0), 2)
            
            # Gözleri işaretle
            cv2.circle(frame, right_point, 5, (0, 0, 0), -1)  
            cv2.circle(frame, left_point, 5, (0, 0, 255), -1) 
            
            # Açıklamaları ekrana yaz
            cv2.putText(frame, f"Roll: {roll_angle:.1f}°", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Head Pose Estimation", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()