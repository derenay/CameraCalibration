import cv2
from ultralytics import YOLO

class DistanceCalculator:
    def __init__(self, model_path, calibration_file, reference_head_width_cm):
        self.model = YOLO(model_path)
        self.reference_head_width = reference_head_width_cm
        self.focal_length = self.load_calibration(calibration_file)

    def load_calibration(self, filename):
        try:
            with open(filename, "r") as f:
                return float(f.read())
        except FileNotFoundError:
            raise ValueError("Kalibrasyon dosyası bulunamadı!")

    def calculate_distance(self, pixel_width):
        return (self.reference_head_width * self.focal_length) / pixel_width

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

    def run(self):
        cap = cv2.VideoCapture(0)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            pixel_width = self.detect_head(frame)
            if pixel_width:
                distance = self.calculate_distance(pixel_width)
                cv2.putText(frame, f"Mesafe: {distance:.2f} cm", (50, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                           
            cv2.imshow("Mesafe Olcumu", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()
