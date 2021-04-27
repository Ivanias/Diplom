from __future__ import print_function
import pyzbar.pyzbar as pyzbar
import numpy as np
import cv2
import func

def decode(image):
    # Нахождение QR-кодов
    decodedObjects = pyzbar.decode(image)
    return decodedObjects

# Отображение местоположения QR-кода
def display(image, decodedObjects):
    # Цикл по всем декодированным объектам
    for decodedObject in decodedObjects:
        points = decodedObject.polygon
        distance=func.getDistance(points)
        print('Distance :', distance)
        angle=func.getAngle(distance,points)
        print('Angle :',angle)
        # Проверка является ли QR-код на расстоянии, допустимом для записи с него информации
        if distance<=1:
            print('Data : ', decodedObject.data, '\n')
        # Рисование ограничивающего прямоугольника
        n = len(points)
        for j in range(0, n):
            cv2.line(image, points[j], points[(j + 1) % n], (255, 0, 0), 3)
    # Отображение результата
    cv2.imshow("Results", image);
    cv2.waitKey(0);

if __name__ == '__main__':
    image = cv2.imread("QR-codeC270distance1m.png")
    decodedObjects = decode(image)
    display(image, decodedObjects)


