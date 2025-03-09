import cv2
from ultralytics import YOLO

class CameraCalibrator:
    def __init__(self, model_path, reference_distance_cm, reference_head_width_cm):
        self.model = YOLO(model_path)
        self.reference_distance = reference_distance_cm
        self.reference_head_width = reference_head_width_cm

    def calibrate(self):
        cap = cv2.VideoCapture(0)
        focal_length = None
        
        print("Kameraya 1 metre uzaklıkta başınızı konumlandırın ve 's' tuşuna basın")
        while True:
            ret, frame = cap.read()
            if not ret:
                raise ValueError("Kamera hatası!")
            
            # Sürekli görüntüyü göster
            cv2.imshow("Kalibrasyon", frame)
            
            # Tuş girişi kontrolü
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):  # 's' tuşuna basıldığında işlem yap
                pixel_width = self.detect_head(frame)
                if pixel_width:
                    focal_length = (pixel_width * self.reference_distance) / self.reference_head_width
                    self.save_calibration(focal_length)
                    print(f"\nKalibrasyon başarılı! Focal length: {focal_length:.2f} kaydedildi.")
                    break
                else:
                    print("\nBaş tespit edilemedi! Lütfen pozisyonunuzu düzeltip tekrar 's' tuşuna basın.")
            
            elif key == ord('q'):  # 'q' ile çıkış
                print("\nKalibrasyon iptal edildi.")
                break
                
        cap.release()
        cv2.destroyAllWindows()
        return focal_length

    def detect_head(self, frame):
        results = self.model(frame)
        boxes = results[0].boxes.xyxy.cpu().numpy()
        
        if boxes.size == 0:
            return None
            
        max_width = 0
        for box in boxes:
            current_width = box[2] - box[0]
            if current_width > max_width:
                max_width = current_width
                
        return max_width if max_width > 0 else None

    def save_calibration(self, focal_length, filename="calibration.txt"):
        with open(filename, "w") as f:
            f.write(str(focal_length))
        print(f"Dosyaya kaydedildi: {filename}")
