import math
import cv2
import mediapipe as mp
import numpy as np

# MediaPipe Face Mesh'i başlat
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Kamera yakalamayı başlat
cap = cv2.VideoCapture(2)


def rotation_matrix_to_angles(rotation_matrix):
    """
    Rotasyon matrisinden Euler açılarını hesaplar.
    :param rotation_matrix: 3x3 rotasyon matrisi
    :return: Derece cinsinden eksenlerdeki açılar
    """
    x = math.atan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
    y = math.atan2(-rotation_matrix[2, 0], math.sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2))
    z = math.atan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
    return np.array([x, y, z]) * 180. / math.pi


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

    # Gerçek dünyadaki yüz koordinatları (örnek değerler)
    face_coordination_in_real_world = np.array([
        [285, 528, 200],
        [285, 371, 152],
        [197, 574, 128],
        [173, 425, 108],
        [360, 574, 128],
        [391, 425, 108]
    ], dtype=np.float64)

    h, w, _ = image.shape

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Yüzdeki belirli anahtar noktaların koordinatlarını al
            face_coordination_in_image = []
            for idx, lm in enumerate(face_landmarks.landmark):
                if idx in [1, 9, 57, 130, 287, 359]:
                    x, y = int(lm.x * w), int(lm.y * h)
                    face_coordination_in_image.append([x, y])

            face_coordination_in_image = np.array(face_coordination_in_image, dtype=np.float64)

            # Kamera matrisi
            focal_length = 1 * w
            cam_matrix = np.array([[focal_length, 0, w / 2],
                                   [0, focal_length, h / 2],
                                   [0, 0, 1]])

            # Distorsiyon matrisi
            dist_matrix = np.zeros((4, 1), dtype=np.float64)

            # solvePnP ile rotasyon vektörünü hesapla
            success, rotation_vec, transition_vec = cv2.solvePnP(
                face_coordination_in_real_world, face_coordination_in_image,
                cam_matrix, dist_matrix)

            # Rodrigues ile rotasyon vektörünü matrise dönüştür
            rotation_matrix, jacobian = cv2.Rodrigues(rotation_vec)

            # Açıları hesapla
            result = rotation_matrix_to_angles(rotation_matrix)

            # Sınırlayıcı kutu için koordinatları hesapla
            all_x = [lm.x * w for lm in face_landmarks.landmark]
            all_y = [lm.y * h for lm in face_landmarks.landmark]
            x_min, x_max = int(min(all_x)), int(max(all_x))
            y_min, y_max = int(min(all_y)), int(max(all_y))

            # Sınırlayıcı kutuyu çiz
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

            # Açıları kutu üzerine yazdır
            for i, info in enumerate(zip(('Pitch', 'Yaw', 'Roll'), result)):
                k, v = info
                text = f'{k}: {int(v)}'
                cv2.putText(image, text, (x_min, y_min + i * 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Görüntüyü göster
    cv2.imshow('Head Pose Angles with Bounding Boxes', image)

    # Çıkış için ESC tuşuna basılmasını kontrol et
    if cv2.waitKey(5) & 0xFF == 27:
        break

# Kaynakları serbest bırak
cap.release()
cv2.destroyAllWindows()