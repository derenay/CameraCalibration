import cv2
import numpy as np
import mediapipe as mp
from ultralytics import YOLO
import open3d as o3d


# YOLOv8 modelini yükle (model yolu ve ağırlıkları uygun şekilde ayarlayın)
yolo_model = YOLO("yolov8n.pt")
yolo_model.to("cuda")  # GPU kullanımı için

# MediaPipe Pose başlatma
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True,
                    model_complexity=1,
                    smooth_landmarks=True,
                    min_detection_confidence=0.8,
                    min_tracking_confidence=0.8)

height_history = {}
# Video kaynağı (kamera veya video dosyası)
cap = cv2.VideoCapture(r"/home/earsal@ETE.local/Desktop/lightweight-human-pose-estimation-3d-demo.pytorch-master/selam.mp4") 

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Tespit listesini her frame başında sıfırla
    detection_list = []

    # YOLOv8 ile tespit yap (sadece insan tespitlerine odaklanıyoruz, COCO'da person class id = 0)
    results = yolo_model(frame, conf=0.5, verbose=False, classes=0)

    boxes = []
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            if cls == 0:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                boxes.append((x1, y1, x2 - x1, y2 - y1))

    for idx, (x, y, w, h) in enumerate(boxes):
        x = max(0, x)
        y = max(0, y)
        crop_img = frame[y:y+h, x:x+w]
        if crop_img.size == 0:
            continue

        crop_rgb = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
        results_pose = pose.process(crop_rgb)

        ear_distance_pixel = None
        ear_distance_3d = None
        body_height_real = None
        info_text = "N/A"

        if results_pose.pose_landmarks:
            landmarks = results_pose.pose_landmarks.landmark

            # 2D eklem noktalarını çizelim
            for lm in landmarks:
                lm_x = int(lm.x * crop_img.shape[1])
                lm_y = int(lm.y * crop_img.shape[0])
                cv2.circle(crop_img, (lm_x, lm_y), 3, (0, 255, 0), -1)

            # Kulaklar (2D ve 3D)
            left_ear = landmarks[mp_pose.PoseLandmark.LEFT_EAR.value]
            right_ear = landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value]

            # Baş ve ayak noktaları
            nose = landmarks[mp_pose.PoseLandmark.NOSE.value]
            left_foot = landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value]
            right_foot = landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value]

            if left_ear.visibility > 0.6 and right_ear.visibility > 0.6:
                lx, ly = int(left_ear.x * crop_img.shape[1]), int(left_ear.y * crop_img.shape[0])
                rx, ry = int(right_ear.x * crop_img.shape[1]), int(right_ear.y * crop_img.shape[0])
                ear_distance_pixel = np.linalg.norm(np.array([lx, ly]) - np.array([rx, ry]))

                if ear_distance_pixel > 0:
                    # 2D kulak arası ölçek faktörü
                    scale_factor_2d = 14.0 / ear_distance_pixel
                    
                    # 3D kulak arası mesafe
                    left_x3d, left_y3d, left_z3d = left_ear.x * crop_img.shape[1], left_ear.y * crop_img.shape[0], left_ear.z * crop_img.shape[1] *0.35
                    right_x3d, right_y3d, right_z3d = right_ear.x * crop_img.shape[1], right_ear.y * crop_img.shape[0], right_ear.z * crop_img.shape[1] *0.35

                    ear_distance_3d = np.linalg.norm(np.array([left_x3d, left_y3d, left_z3d]) -
                                                     np.array([right_x3d, right_y3d, right_z3d]))

                    if ear_distance_3d > 0:
                        scale_factor_3d = 14.0 / ear_distance_3d
                    else:
                        scale_factor_3d = scale_factor_2d

                    # 3D kulak mesafesi ölçeklenmiş hali
                    ear_distance_3d_real = scale_factor_3d * ear_distance_3d
                    # print(f"{ear_distance_pixel}   {ear_distance_3d}")
                    # BOY HESAPLAMA
                    if nose.visibility > 0.5 and left_foot.visibility > 0.5 and right_foot.visibility > 0.5:
                        head_x3d, head_y3d, head_z3d = nose.x * crop_img.shape[1], nose.y * crop_img.shape[0], nose.z * crop_img.shape[1]
                        foot_x3d, foot_y3d, foot_z3d = ((left_foot.x + right_foot.x) / 2) * crop_img.shape[1], \
                                                       ((left_foot.y + right_foot.y) / 2) * crop_img.shape[0], \
                                                       ((left_foot.z + right_foot.z) / 2) * crop_img.shape[1]

                        # Baş ve ayak arasındaki 3D mesafe
                        body_height_3d = np.linalg.norm(np.array([head_x3d, head_y3d, head_z3d]) -
                                                        np.array([foot_x3d, foot_y3d, foot_z3d]))

                        # Gerçek boy (cm cinsinden)
                        
                        body_height_real = scale_factor_3d * body_height_3d
                        if 13.5>body_height_3d / ear_distance_3d >11:
                            if idx in height_history:
                                height_history[idx].append(body_height_real)
                            else:
                                height_history[idx] = [body_height_real]
                            avg_height = np.mean(height_history[idx])
                           
                            print(f"{idx}  {body_height_3d}    {ear_distance_3d}   {avg_height}")
                            info_text = f"Boy: {avg_height}cm" 
                            
          
                    

        # Görüntüye bilgileri ekleyelim
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, f"{idx}: {info_text}", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        frame[y:y+h, x:x+w] = crop_img

        # Tespit bilgileri listeye ekleyelim
        detection_info = {
            "detection_id": idx,
            "box": (x, y, w, h),
            "ear_distance_pixel": ear_distance_pixel,
            "ear_distance_3d_cm": 14.0,
            "body_height_cm": body_height_real
        }
        detection_list.append(detection_info)




    cv2.imshow("YOLOv8 + MediaPipe Pose (2D -> 3D) - Boy Hesaplama", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
