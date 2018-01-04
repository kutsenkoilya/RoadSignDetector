# RoadSignDetector

Детектор дорожных знаков

Файлы и папки:
/_verification - картинки на которых распознаются знаки

/camer_cal - картинки для создания калибровочной матрицы

01_straigthLaneDetection.py - скрипт для распознавания прямой разметки на видео

02_cameraCalibrationPickle.py - скрипт для создания калибровочной матрицы для распознавания изогнутой разметки

02_rectangleLaneDetection.py - скрипт для распознавания изогнутой разметки

03_imageSignRecognizer.py - скрипт для распознавания знаков на картинках из папки _verification/pic/roadswsigns

03_webCamSignRecognizer.py - скрипт для распознвания знаков на видео вебакмеры

camera_matrix.pkl - огурчик, содержащий калибровочную матрицу

camera_matrix_640x480.pkl огурчик, содержащий калибровочную матрицу для изображения 640х480

lib_squareSignDetector.py - метод распознавания квадратных знаков

lib_roundSignDetector.py - метод распознавания круглых знаков (работает как квадратные)

