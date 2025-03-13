from kalibrasyon import CameraCalibrator
from deneme02 import DistanceCalculator

def calibrate():
    calibrator = CameraCalibrator(
        model_path="calibration/yolo11n.pt",  
        reference_distance_cm=40,
        reference_head_width_cm=15
    )

    calibrator.calibrate()



def start():

    calculator = DistanceCalculator(
        model_path="calibration/yolo11n.pt",  
        calibration_file='calibration/calibration.txt',
        reference_head_width_cm=15
    )

    calculator.run()
    
#calibrate()
start()

"""
Amacım kameranın içsel parametlereinden çektiği alandaki birebir cm olarak uyuştuğu kısmı bulup,
bu kısımdan bir referans noktası almam lazım pixel-gerçek yüz genişliği olaraktan.

Sonra bu bulduğum referans değerini kullanıp tespitte bulduğum insanların kafa tespininden insanlar arasındaki uzunluğu, insan ununluğu bulmayı hedefliyorum.

Sonrasında bunu object detection modeline implemente edicez:
    Nasıl olucak?:
        çantayıda referans olarak bir kez belirleyip konum tespiti ile işlem yapabiliriz şöyle demek istiyorum eğer insan çantanın çok arkasında ve çanta insanla yan yana gibi gözüküoyrsa kamerada bunu engellememiz lazım 
        bunuda çantanın bağladığımız insanla arasındaki mesafayi hem merkez kordinatlarından hemde bu bulacağımız herçek uzaklıkları arasındaki mesafeden bulmayı hedefliyorum bu sayede
        3D görüntü tespiti yapabilecek konuma gelicez tahminimce sadece listelerde tutmamız lazım bu verileri ki kullanıcı trackerden yeni bir id alırsa işimizi bozmasın
        

"""