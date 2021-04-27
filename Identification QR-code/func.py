import numpy as np
import cv2
import configparser

config = configparser.ConfigParser()  # создаём объекта парсера
config.read("settings.ini")  # читаем конфиг



def getDistance (points):
    x1 = int(str(points[0])[8:str(points[0]).find(",")])
    x2 = int(str(points[2])[8:str(points[2]).find(",")])
    sizePX = x2 - x1
    sizeM=float(config["QR-code"]["sizeM"])
    focus=int(config["QR-code"]["focus"])
    distance = sizeM * focus / sizePX
    return distance

def getAngle (distance,points):
    # Находим координаты точки центра QR-кода
    x11 = int((str(points[3]))[8:str(points[3]).find(",")])
    x12 = int((str(points[2]))[8:str(points[2]).find(",")])
    x21 = int((str(points[0]))[8:str(points[0]).find(",")])
    x22 = int((str(points[1]))[8:str(points[1]).find(",")])
    y11 = int((str(points[3]))[(str(points[3]).find("y="))+2:-1])
    y12 = int((str(points[2]))[(str(points[2]).find("y="))+2:-1])
    y21 = int((str(points[0]))[(str(points[0]).find("y="))+2:-1])
    y22 = int((str(points[1]))[(str(points[1]).find("y="))+2:-1])
    coordinates=[int(((x11 + x12) / 2 + (x21 + x22) / 2) / 2), int(((y11 + y21) / 2 + (y12 + y22) / 2) / 2)]
    point11 = [x11,y11]
    point12 = [x12, y12]
    point21 = [x21, y21]
    point22 = [x22, y22]
    # Находим координаты точки центра изображения
    center=[int(config["QR-code"]["width"])/2,int(config["QR-code"]["height"])/2]
    disPoint11Center=np.hypot(abs(coordinates[0]-point11[0]),abs(coordinates[1] -point11[1]))
    disPoint12Center=np.hypot(abs(coordinates[0]-point12[0]),abs(coordinates[1] -point12[1]))
    disPoint21Center=np.hypot(abs(coordinates[0]-point21[0]),abs(coordinates[1] -point21[1]))
    disPoint22Center=np.hypot(abs(coordinates[0]-point22[0]),abs(coordinates[1] -point22[1]))
    ratioPXinM=(np.hypot((float(config["QR-code"]["sizeM"])),(float(config["QR-code"]["sizeM"])))/2)/((disPoint11Center+disPoint12Center+disPoint21Center+disPoint22Center)/4)
    # Находим расстояние от центра изображения до центра QR-кода (в пикселях)
    differenceX=abs(center[0]-coordinates[0])
    differenceY = abs(center[1] - coordinates[1])
    deviationPX=np.hypot(differenceX,differenceY)
    # Находим расстояние от центра изображения до центра QR-кода (в метрах)
    deviationM=deviationPX*ratioPXinM
    # Находим угол
    sinAngle=deviationM/distance
    angle = (np.arcsin(sinAngle) / np.pi) * 180
    return angle




