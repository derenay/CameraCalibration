#You need to import files to here

calibrator = CameraCalibrator(
    model_path='yolov8n.pt',  # Model yolunuzu girin
    reference_distance_cm=60,
    reference_head_width_cm=15
)

calibrator.calibrate()



calculator = DistanceCalculator(
    model_path='yolov8n.pt',  # Model yolunuzu girin
    calibration_file='calibration.txt',
    reference_head_width_cm=15
)

calculator.run()
