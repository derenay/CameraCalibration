import cv2
import mediapipe as mp
import math

# MediaPipe Face Mesh'i başlat
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Kamera yakalamayı başlat
cap = cv2.VideoCapture(2)

def calculate_distance(point1, point2):
    """
    İki nokta arasındaki Öklid mesafesini hesaplar.
    :param point1: İlk nokta (x1, y1)
    :param point2: İkinci nokta (x2, y2)
    :return: İki nokta arasındaki mesafe
    """
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Kamera görüntüsü alınamadı.")
        continue

    # Görüntüyü RGB formatına dönüştür
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image)

    # Görüntüyü tekrar BGR formatına dönüştür
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    h, w, _ = image.shape

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Kulak arkası noktalarını al (234 ve 454)
            left_ear_point = (int(face_landmarks.landmark[127].x * w), int(face_landmarks.landmark[234].y * h))
            right_ear_point = (int(face_landmarks.landmark[264].x * w), int(face_landmarks.landmark[454].y * h))

            # İki nokta arasındaki mesafeyi hesapla
            face_width = calculate_distance(left_ear_point, right_ear_point)

            # Mesafeyi ekrana yazdır
            cv2.putText(image, f"Face Width: {int(face_width)} px", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # Kulak arkası noktalarını işaretlemek için çember çiz
            cv2.circle(image, left_ear_point, 5, (0, 255, 0), -1)
            cv2.circle(image, right_ear_point, 5, (0, 255, 0), -1)

            # İki nokta arasına bir çizgi çiz
            cv2.line(image, left_ear_point, right_ear_point, (255, 0, 0), 2)

    # Görüntüyü göster
    cv2.imshow('Face Width Measurement', image)

    # Çıkış için ESC tuşuna basılmasını kontrol et
    if cv2.waitKey(5) & 0xFF == 27:
        break

# Kaynakları serbest bırak
cap.release()
cv2.destroyAllWindows()