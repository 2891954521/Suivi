import os
import time
import threading

import cv2
import requests
import numpy as np

#
class Person:

    def __init__(self, uid, gray, x, y, w, h):
        self.uid = uid

        self.image = gray

        # 目标在某一帧是否被检测到
        self.isFind = True

        # 目标从视野中丢失的次数
        self.lostTime = 0

        # 标记一下新出现在视野中的人
        self.setColor((0, 255, 0))

        self.updateLocation(x, y, w, h)

    # 设置颜色，指定帧数过后恢复
    def setColor(self, color:tuple, time:int = 10):
        self.color = color
        self.colorTime = time
    
    # 更新坐标
    def updateLocation(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h

    # 更新目标的坐标
    def updatePerson(self, image):

        self.isFind = True

        # 如果目标是重新出现在视野中的
        if self.lostTime > 0:
            # 改变一下框的颜色
            self.lostTime = 0
            self.setColor((0, 0, 255))

        if self.colorTime > 0:
            self.colorTime -= 1
            cv2.rectangle(image, (self.x, self.y), (self.x + self.w, self.y + self.h), self.color, 2)
        else:
            cv2.rectangle(image, (self.x, self.y), (self.x + self.w, self.y + self.h), (255, 0, 0), 2)

        cv2.putText(image, "p" + str(self.uid), (self.x + 5, self.y + 5),  cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)


class Stranger(Person):
    
    def __init__(self, uid, x, y, w, h):
        super().__init__(uid, None, x, y, w, h)

        self.count = 0

        self.notfoundTime = 0

        self.faces = []

        self.time = [0 for _ in range(5)]

    # 简单判断是否是同一个目标
    def isPersion(self, x, y, w, h) -> bool:

        if self.x > x + w or self.y > y + h or self.x + self.w < x or self.y + self.h < y:
            return False

        colInt = abs(min(self.x + self.w, x + w) - max(self.x, x))
        rowInt = abs(min(self.y + self.h, y + h) - max(self.y, y))
        overlap_area = colInt * rowInt
        area1 = self.w * self.h
        area2 = w * h

        # 目标移动范围在合理区间内即认为是同一个目标
        if overlap_area / (area1 + area2 - overlap_area) > 0.3:
            self.updateLocation(x, y, w, h)
            return True
        else:
            return False

    # 对陌生人进行识别
    # 返回值：1：识别完成，0：还在识别，-1：超时
    def recognitionStranger(self, gray) -> int:

        face = gray[self.y : self.y + self.h, self.x : self.x + self.w]

        if len(self.faces) < 5:
            # 先捕获5张人脸
            self.faces.append(face)

            if len(self.faces) == 5:
                # 用这5张人脸训练一个临时模型
                self.recognizer = cv2.face.LBPHFaceRecognizer_create()
                self.recognizer.train(self.faces, np.array([i for i in range(5)]))

        elif self.count < 10:
            # 用这个模型匹配10张人脸
            i, confidence = self.recognizer.predict(face)

            if 0 < confidence < 45:
                self.time[i] += 1
                self.count += 1
            else:
                self.notfoundTime += 1
                if self.notfoundTime > 20:
                    return -1
            
            if self.count == 10:
                # 取最优匹配的人脸
                self.image = self.faces[self.time.index(max(self.time))]

                path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'face')
                
                cv2.imwrite(os.path.join(path, str(self.uid) + ".jpg"), self.image)

                # 释放内存
                del self.faces
                return 1

        return 0

    # 更新目标的坐标
    def updatePerson(self, image):

        self.isFind = True

        # 如果目标是重新出现在视野中的
        if self.lostTime > 0:
            # 改变一下框的颜色
            self.lostTime = 0
            self.setColor((0, 0, 255))

        if self.colorTime > 0:
            self.colorTime -= 1
            cv2.rectangle(image, (self.x, self.y), (self.x + self.w, self.y + self.h), self.color, 2)
        else:
            cv2.rectangle(image, (self.x, self.y), (self.x + self.w, self.y + self.h), (0, 255, 0), 2)

        cv2.putText(image, str(self.uid), (self.x + 5, self.y + 5),  cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)


class RecognitionServer(threading.Thread):

    def __init__(self):
        super().__init__()

        # 一帧图像
        self.frame = None

        # 视野中未出现人的次数
        self.notFoundTime = 0

        # 存放所有已经找到的人
        self.strangers = {}
        self.persons = {}

        self.isMoving = False

        # 进程是否结束
        self.isFinish = False

        # 是否需要更新模型
        self.needUpdate = False

        self.camera = cv2.VideoCapture(0)

        self.condition = threading.Condition()

        module = os.path.join(os.path.dirname(os.path.abspath(__file__)), './modules/haarcascade_frontalface_default.xml')

        if os.path.exists(module):
            self.faceCascade = cv2.CascadeClassifier(module)
        else:
            raise RuntimeError('模型文件不存在:' + module)

    def run(self):

        while True:

            if self.isFinish:
                break

            image = cv2.flip(self.camera.read()[1], 1)

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            faces = self.faceCascade.detectMultiScale(
                gray,
                scaleFactor=1.2,
                minNeighbors=3,
                minSize=(50, 50)
            )

            if len(faces) == 0:
                if self.notFoundTime > 500:
                    cv2.waitKey(1000)
                else:
                    self.notFoundTime += 1
                    cv2.waitKey(100)
            else:
                self.notFoundTime = 0

                # if len(faces) == 1:

                #     (x, y, w, h) = faces[0]

                #     height = image.shape[0]
                #     width = image.shape[1]

                #     if x > width / 2:
                #         self.moveRight()
                #     elif width / 2 > x + w:
                #         self.moveLeft()
                #     elif y > height / 2:
                #         self.moveDown()
                #     elif height / 2 > y + h:
                #         self.moveUp()

                # 遍历人脸找已知的目标
                for (x, y, w, h) in faces:
                    
                    if len(self.persons) > 0:
                        # 检查有没有已知的目标
                        uid, confidence = self.recognizer.predict(gray[y : y + h, x : x + w])

                        if 0 < confidence < 45:
                            self.persons[uid].updateLocation(x, y, w, h)
                            self.persons[uid].updatePerson(image)
                            continue

                    # 遍历所有未知目标
                    for (uid, stranger) in self.strangers.items():
                        if not stranger.isFind and stranger.isPersion(x, y, w, h):
                            stranger.updatePerson(image)
                            code = stranger.recognitionStranger(gray)
                            if code == 1:
                                # 对陌生人的识别结束，将其添加到已知目标列表
                                print("find new persion:" + str(uid))
                                self.persons[uid] = self.strangers.pop(uid)
                                self.needUpdate = True
                            elif code == -1:
                                self.strangers.pop(uid)
                            break
                    else:
                        # 都找不到就添加一个新目标
                        uid = int(time.time() * 100) % 100000
                        self.strangers[uid] = Stranger(uid, x, y, w, h)

                self.removeUnfindBody()

                if self.needUpdate:
                    self.recognizer = cv2.face.LBPHFaceRecognizer_create()
                    self.recognizer.train(list(map(lambda person:person.image, self.persons.values())), np.array(list(map(lambda person:person.uid, self.persons.values()))))
                    self.needUpdate = False

                cv2.waitKey(50)

            with self.condition:
                self.frame = image
                self.condition.notifyAll()

    # 移除已丢失的目标
    def removeUnfindBody(self):

        for uid in list(self.persons.keys()):
            person = self.persons[uid]
            if not person.isFind:
                # 目标丢失
                person.lostTime += 1
                # 丢失次数过多后删除目标
                if person.lostTime == 3:
                    self.persons.pop(uid)
            else:
                person.isFind = False

        for uid in list(self.strangers.keys()):
            stranger = self.strangers[uid]
            if not stranger.isFind:
                # 目标丢失
                stranger.lostTime += 1
                # 丢失次数过多后删除目标
                if stranger.lostTime == 50:
                    self.strangers.pop(uid)
            else:
                stranger.isFind = False
    
    def moveUp(self):
        if not self.isMoving:
            self.isMoving = True
            MoveCamera(self, 'http://192.168.5.4/up').start()
    
    def moveDown(self):
        if not self.isMoving:
            self.isMoving = True
            MoveCamera(self, 'http://192.168.5.4/down').start()

    def moveLeft(self):
        if not self.isMoving:
            self.isMoving = True
            MoveCamera(self, 'http://192.168.5.4/left').start()

    def moveRight(self):
        if not self.isMoving:
            self.isMoving = True
            MoveCamera(self, 'http://192.168.5.4/right').start()


# 用于移动摄像头的线程
class MoveCamera(threading.Thread):

    def __init__(self, parent:RecognitionServer, url:str):
        super().__init__()
        self.parent = parent
        self.url = url

    def run(self):
        try:
            requests.get(self.url)
        except:
            pass
        self.parent.isMoving = False
            
