import math
import cv2
import numpy as np
from ultralytics import YOLO
import mediapipe as mp

# MediaPipe Face Detection ve Face Mesh'i başlat
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.2)

# YOLO modelini yükle
main_model = YOLO("yolov8n.pt")

# Kamera parametreleri (kalibrasyonla bulunmalıdır)
real_head_size = 17  # Gerçek dünyada bir insanın kafasının boyu (cm)
pixel = 130
real_distance = 50
focal_length_cm = (pixel * real_head_size) / real_distance
focal_length_mm = focal_length_cm * 10

# Kamerayı aç
cap = cv2.VideoCapture(0)

# Rotasyon matrisini açıya dönüştürme fonksiyonu
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



def calculate_distance(real_head_size, focal_length_mm, image_head_size, angles):
    """
    Kafa açılarından (pitch, yaw, roll) etkilenerek mesafeyi hesaplar.
    :param real_head_size: Gerçek kafa boyutu (cm)
    :param focal_length_mm: Kamera odak uzaklığı (mm)
    :param image_head_size: Görüntüdeki kafa boyutu (piksel)
    :param angles: Başın açılarının listesi [pitch, yaw, roll] (derece cinsinden)
    :return: Mesafe (cm)
    """

    # Açıları radyana dönüştür
    pitch = math.radians(angles[0])  # Yukarı-aşağı eğilme açısı
    yaw = math.radians(angles[1])    # Sağa-sola dönüş açısı
    roll = math.radians(angles[2])   # Eğilme açısı

    # Görüntüdeki baş boyutunun etkisini düzeltmek için açıları kullanıyoruz

    
    corrected_image_head_size = (image_head_size / (math.cos(pitch)))

    # Mesafeyi hesapla (açıları hesaba katarak)
    distance = (real_head_size * focal_length_mm) / corrected_image_head_size
    
    return distance



import math

def height_calculator(distance, box_height_pixels, image_head_size, angles, x1, x2, y1, y2):
    """
    İnsan yüksekliğini hesaplayan fonksiyon.
    :param distance: İnsanın kameraya olan gerçek mesafesi (cm).
    :param image_head_size: Görüntüdeki baş boyutu (piksel cinsinden).
    :param angles: Başın açılarının listesi [pitch, yaw, roll] (derece cinsinden).
    :param x1, x2, y1, y2: Bounding box'un koordinatları (piksel cinsinden).
    :param focal_length_mm: Kameranın odak uzaklığı (mm).
    :return: Kişinin tahmini boyu (cm).
    """

    # Gerçek dünyadaki baş boyutu (cm)
    real_head_size = 17

    # Bounding box'un yüksekliği (piksel cinsinden)
     

    # Açıları radyana dönüştür
    pitch = math.radians(angles[0])  # Yukarı-aşağı eğilme açısı
    yaw = math.radians(angles[1])    # Sağa-sola dönüş açısı

    real_length_2 = real_head_size * (box_height_pixels / pixel) * (distance / real_distance)
    # person_height_cm = (real_head_size*box_height_pixels)/image_head_size
    return real_length_2

frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Performans için her üçüncü frame'i işle
    if frame_count % 2 == 0:
        # YOLO ile insan tespiti
        results = main_model.track(frame, persist=True, verbose=False)
        
        if not results or results[0] is None or results[0].boxes.id is None:
            continue
        
        boxes = results[0].boxes.xywh.cpu().numpy()
        cls = results[0].boxes.cls.cpu().numpy()
        track_ids = results[0].boxes.id.int().cpu().tolist()  
        names = results[0].names
       
        persons = [(box, tid) for box, tid, cl in zip(boxes, track_ids, cls) if cl == 0]
    
        for (box, track_id) in persons:
            x, y, w, h = box            
            x1 = int(x - w / 2)
            y1 = int(y - h / 2)
            x2 = int(x + w / 2)
            y2 = int(y + h / 2)

            # ROI'yi kırparak yüz tespiti yap
            roi = frame[y1:y2, x1:x2].copy()
            if roi.size is None:
                continue
            
            rgb_frame = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            head_results = face_detection.process(rgb_frame)
            
             
            if head_results.detections is None:
                continue
            
            # İlk yüz tespitini al
            detection = head_results.detections[0]
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = roi.shape
            a1_roi = int(bboxC.xmin * iw)
            b1_roi = int(bboxC.ymin * ih)
            a2_roi = int((bboxC.xmin + bboxC.width) * iw)
            b2_roi = int((bboxC.ymin + bboxC.height) * ih)

            # Orijinal görüntüye göre yüz koordinatlarını hesapla
            face_x1 = x1 + a1_roi
            face_y1 = y1 + b1_roi
            face_x2 = x1 + a2_roi
            face_y2 = y1 + b2_roi

            # Yüzün merkezine bir nokta koy
            cv2.rectangle(frame, (face_x1, face_y1), (face_x2, face_y2), (255, 0, 0), 2)
            cv2.circle(frame, ((face_x1 + face_x2) // 2, (face_y1 + face_y2) // 2), 2, (0, 0, 255), -1)
            
            # Piksel cinsinden kafa boyutu
            image_head_size = face_y2 - face_y1

            # Mesafe hesaplama
            if image_head_size > 0:  # Sıfıra bölme hatası önlemek için kontrol
                

                face_region = frame[face_y1:face_y2, face_x1:face_x2] if face_y1 < face_y2 and face_x1 < face_x2 else None

                # Eğer yüz bölgesi geçerliyse RGB'ye dönüştürme
                if face_region is not None and face_region.size > 0:
                    try:
                        face_region_rgb = cv2.cvtColor(face_region, cv2.COLOR_BGR2RGB)
                    except cv2.error as e:
                        print(f"Hata: {e}")
                        face_region_rgb = None
                else:
                    print("ben çalıştım")
                    continue
                
                
                face_results = face_mesh.process(face_region_rgb)

                if face_results.multi_face_landmarks:
                    for face_landmarks in face_results.multi_face_landmarks:
                        # Yüzdeki belirli anahtar noktaların koordinatlarını al
                        face_coordination_in_image = []
                        for idx, lm in enumerate(face_landmarks.landmark):
                            if idx in [1, 9, 57, 130, 287, 359]:  # Önemli yüz noktaları
                                x_roi = int(lm.x * (face_x2 - face_x1))  # Yüz bölgesi genişliği
                                y_roi = int(lm.y * (face_y2 - face_y1))  # Yüz bölgesi yüksekliği
                                # Orijinal görüntüye göre koordinatları hesapla
                                x = face_x1 + x_roi
                                y = face_y1 + y_roi
                                face_coordination_in_image.append([x, y])
                                
                        face_coordination_in_image = np.array(face_coordination_in_image, dtype=np.float64)

                        # Gerçek dünyadaki yüz koordinatları (örnek değerler)
                        face_coordination_in_real_world = np.array([
                            [285, 528, 200],
                            [285, 371, 152],
                            [197, 574, 128],
                            [173, 425, 108],
                            [360, 574, 128],
                            [391, 425, 108]
                        ], dtype=np.float64)

                        # Kamera matrisi
                        h, w, _ = frame.shape
                        cam_matrix = np.array([[focal_length_mm, 0, w / 2],
                                               [0, focal_length_mm, h / 2],
                                               [0, 0, 1]])

                        # Distorsiyon matrisi
                        dist_matrix = np.zeros((4, 1), dtype=np.float64)

                        # solvePnP ile rotasyon vektörünü hesapla
                        success, rotation_vec, transition_vec = cv2.solvePnP(
                            face_coordination_in_real_world, face_coordination_in_image,
                            cam_matrix, dist_matrix)

                        if success:
                            # Rodrigues ile rotasyon vektörünü matrise dönüştür
                            rotation_matrix, jacobian = cv2.Rodrigues(rotation_vec)
                            
                            # Açıları hesapla
                            angles = rotation_matrix_to_angles(rotation_matrix)
                            distance = calculate_distance(real_head_size, focal_length_mm, image_head_size, angles)
                            person_height = height_calculator(distance, y2-y1,image_head_size, angles, x1, x2, y1, y2)
            
                            # Kafayı çerçeve içine alma ve mesafeyi yazdırma
                            
                            cv2.rectangle(frame, (face_x1, face_y1), (face_x2, face_y2), (0, 0, 255), 2)
                            cv2.putText(
                                frame,
                                f"{person_height:.2f} cm | Distance: {distance:.2f}cm ",
                                (face_x1, face_y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1,
                                (0, 0, 255),
                                2,
                            )
                            cv2.putText(
                                frame,
                                f"Pitch: {int(angles[0])}, Yaw: {int(angles[1])}, Roll: {int(angles[2])}",
                                (face_x1, face_y1 - 30),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,
                                (0, 0, 255),
                                2,
                            )

    frame_count += 1

    # Görüntüyü göster
    cv2.imshow("Head Pose Estimation with YOLO", frame)

    # Çıkış için 'q' tuşuna basılmasını kontrol et
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Kaynakları serbest bırak
cap.release()
cv2.destroyAllWindows()
