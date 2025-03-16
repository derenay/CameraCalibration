import math
import numpy as np

def rotation_matrix(roll, pitch, yaw):
    """Euler açılarından dönüş matrisini oluşturur (ZYX sırası)"""
    # Roll (Z), Pitch (Y), Yaw (X) - MediaPipe genellikle bu sırayı kullanır
    # https://developers.google.com/mediapipe/solutions/vision/face_landmarker#coordinates
    cr = np.cos(roll)
    sr = np.sin(roll)
    cp = np.cos(pitch)
    sp = np.sin(pitch)
    cy = np.cos(yaw)
    sy = np.sin(yaw)
    
    return np.array([
        [cy*cr + sy*sp*sr, -cy*sr + sy*sp*cr, sy*cp],
        [sr*cp, cr*cp, -sp],
        [-sy*cr + cy*sp*sr, sy*sr + cy*sp*cr, cy*cp]
    ])

def calculate_real_height(l_head_real, l_head_pixel, h_person_pixel, pitch, yaw, roll):
    """
    Kafa açılarını kullanarak gerçek boyu hesaplar
    
    Parametreler:
    - l_head_real: Gerçek kafa uzunluğu (cm)
    - l_head_pixel: Görüntüde kafa uzunluğu (piksel)
    - h_person_pixel: Görüntüde tüm vücut yüksekliği (piksel)
    - pitch, yaw, roll: Radyan cinsinden açılar (MediaPipe'den direkt)
    """
    # 1. Kafa dönüş matrisini oluştur
    R = rotation_matrix(roll, pitch, yaw)
    
    # 2. Kafanın yerel koordinatlarda üst yön vektörü (burundan alına)
    head_top_vec = np.array([0, -l_head_real, 0])  # Y ekseni MediaPipe'de aşağı yön
    
    # 3. Vektörü döndür
    rotated_vec = R @ head_top_vec
    
    # 4. Perspektif projeksiyon (basitleştirilmiş, kamera parametreleri yok)
    projected_length = math.sqrt(rotated_vec[0]**2 + rotated_vec[1]**2)
    
    # 5. Ölçek faktörünü hesapla
    scale_factor = projected_length / l_head_real
    
    # 6. Gerçek boyu hesapla
    pixel_per_cm = l_head_pixel / (l_head_real * scale_factor)
    real_height_cm = h_person_pixel / pixel_per_cm
    
    return real_height_cm

# Örnek Kullanım
if __name__ == "__main__":
    # MediaPipe'den gelen açılar (radyan)
    pitch = math.radians(15)  # Yukarı bakıyor
    yaw = math.radians(30)   # Sağa dönük
    roll = math.radians(-10) # Hafif sola yatık
    
    real_height = calculate_real_height(
        l_head_real=22,       # Ortalama kafa uzunluğu
        l_head_pixel=45,      # Görüntüde ölçülen kafa yüksekliği
        h_person_pixel=350,    # Görüntüde ölçülen tüm vücut
        pitch=pitch,
        yaw=yaw,
        roll=roll
    )
    
    print(f"Tahmini Gerçek Boy: {real_height:.1f} cm")
