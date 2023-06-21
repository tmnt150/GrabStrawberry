import cv2
import time
import mediapipe as mp
import random
import math
import numpy as np
import pygame

pygame.mixer.init()
mpDraw = mp.solutions.drawing_utils
mpHands = mp.solutions.hands
hands = mpHands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# 載入物件圖(白邊可去背)
ObjSizeX, ObjSizeY = 80, 80
BucketSizeX, BucketSizeY = 180, 80
Obj = cv2.resize(cv2.imread('m.jpg'), (ObjSizeX, ObjSizeY))  # 物件圖
# 載入籃子圖
Bucket = cv2.resize(cv2.imread('b.jpg'), (BucketSizeX, BucketSizeY))  # 物件圖
# 抓取狀態
catchObj = False
# 是否產生新物件
NewObj = True
# Obj起始位置
x, y = 0, 0
# 得分
Score = 0

AngleTH = 130  # 判斷張開角度


def findAngleF(a, b, c):
    ang = math.degrees(math.atan2(c[2] - b[2], c[1] - b[1]) - math.atan2(a[2] - b[2], a[1] - b[1]))
    if ang < 0:
        ang = ang + 360
    if ang >= 360 - ang:
        ang = 360 - ang
    return round(ang, 2)


# 繪圖：將物件放置在背景的特定位置，白邊去背
def addPng(ax, ay, bg, png):
    global NewObj
    pngY, pngX, channels = png.shape
    bgY, bgX, channels = bg.shape
    if (ax + pngX <= bgX and ax >= 0) and (ay + pngY <= bgY and ay >= 0):
        roi = bg[ay:ay + pngY, ax:ax + pngX]  # 放置位置
        # 1.建立png黑色部份作為遮罩
        pnggray = cv2.cvtColor(png, cv2.COLOR_BGR2GRAY)
        ret, mask = cv2.threshold(pnggray, thresh=230, maxval=255, type=cv2.THRESH_BINARY)  # 小於thresh變0，其餘變成maxval
        mask_inv = cv2.bitwise_not(mask)  # 反向
        # cv2.imshow('mask',mask_inv)
        # cv2.waitKey(0)
        # 2.找出未來貼上的背景部份
        bg_roi = cv2.bitwise_and(roi, roi, mask=mask)  # 擷取出剩下的背景
        # cv2.imshow('mask',bg_roi)
        # cv2.waitKey(0)
        # 3.取出Logo要顯示部份
        png_roi = cv2.bitwise_and(png, png, mask=mask_inv)  # 取出真正要顯示的部份
        # cv2.imshow('mask',png_roi)
        # cv2.waitKey(0)
        # 4.將以上部份組合
        dst = cv2.add(bg_roi, png_roi)
        bg[ay:ay + pngY, ax:ax + pngX] = dst
        return bg
    else:
        NewObj = True
        return bg


# 檢測手掌狀態
def fist(img):
    imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # 轉換色彩頻道
    h, w, c = frame.shape  # 取得螢幕長寬色彩
    results = hands.process(imgRGB)  # 手部辨識結果

    if results.multi_hand_landmarks:  # 如果有找到手部
        for i in range(len(results.multi_handedness)):  # 所有的手
            thisHandType = results.multi_handedness[i].classification[0].label  # 手的屬性
            thisHand = results.multi_hand_landmarks[i]  # 取得這隻手
            mpDraw.draw_landmarks(frame, thisHand, mpHands.HAND_CONNECTIONS)  # 利用工具畫
            # 學習自己畫關節(了解關節座標位置)
            thisHandLMList = []
            for id, lm in enumerate(thisHand.landmark):  # id=編號,lm=座標
                thisHandLMList.append([id, lm.x, lm.y, lm.z])
                hx, hy = int(lm.x * w), int(lm.y * h)  # 計算座標
                cv2.circle(frame, (hx, hy), 5, (255, 0, 0), cv2.FILLED)  # 在關節點上標藍色圓形
                cv2.putText(frame, str(id), (hx, hy), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 255), 1)
                if id == 0:
                    cv2.putText(frame, thisHandType, (hx, hy - 30), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
            finger = [0, 0, 0, 0, 0]
            if (findAngleF(thisHandLMList[0], thisHandLMList[3], thisHandLMList[4]) > AngleTH):
                finger[0] = 1
            if (findAngleF(thisHandLMList[0], thisHandLMList[6], thisHandLMList[8]) > AngleTH):
                finger[1] = 1
            if (findAngleF(thisHandLMList[0], thisHandLMList[10], thisHandLMList[12]) > AngleTH):
                finger[2] = 1
            if (findAngleF(thisHandLMList[0], thisHandLMList[14], thisHandLMList[16]) > AngleTH):
                finger[3] = 1
            if (findAngleF(thisHandLMList[0], thisHandLMList[18], thisHandLMList[20]) > AngleTH):
                finger[4] = 1
            # print(finger)

            totalFingers = finger.count(1)
            x1, y1 = np.amin(np.array(thisHandLMList)[:, 1]), np.amin(np.array(thisHandLMList)[:, 2])
            x2, y2 = np.amax(np.array(thisHandLMList)[:, 1]), np.amax(np.array(thisHandLMList)[:, 2])
            return totalFingers, x1, y1, x2, y2
    else:
        return None, None, None, None, None


cap = cv2.VideoCapture(0)
while cap.isOpened():
    stime = time.time()
    ret, frame = cap.read()
    h, w, c = frame.shape
    frame = cv2.flip(frame, 1)

    totalFingers, hx1, hy1, hx2, hy2 = fist(frame)
    if not (totalFingers == None):
        if totalFingers <= 1:
            if ((x + ObjSizeX // 2) >= hx1 * w and (x + ObjSizeX // 2) <= hx2 * w and (
                    y + ObjSizeY // 2) >= hy1 * h and (y + ObjSizeY // 2) <= hy2 * h):
                catchObj = True
            else:
                catchObj = False
        else:
            catchObj = False
    else:
        catchObj = False

    BucketX = round((w - BucketSizeX) / 2)
    BucketY = round(h - BucketSizeY / 2 - 50)
    frame = addPng(BucketX, BucketY, frame, Bucket)

    if catchObj:
        x, y = round(((hx1 + hx2) * w - ObjSizeX) // 2), round(((hy1 + hy2) * h - ObjSizeY) // 2)
        frame = addPng(x, y, frame, Obj)
        if ((x + ObjSizeX // 2) >= BucketX and (x + ObjSizeX // 2) <= BucketX + BucketSizeX and (
                y + ObjSizeY // 2) >= BucketY and (y + ObjSizeY // 2) <= BucketY + BucketSizeY):
            Score = Score + 1
            print("Score=" + str(Score))
            NewObj = True
            catchObj = False
            pygame.mixer.music.load("eat.mp3")
            pygame.mixer.music.play()

            # 重設座標
            x, y = random.randint(10, w - ObjSizeX - 10), 5
    elif (NewObj == False):
        y = y + 5
        frame = addPng(x, y, frame, Obj)
    else:
        x, y = random.randint(10, w - ObjSizeX - 10), 5
        NewObj = False
    etime = time.time()
    fps = round(1 / (etime - stime), 2)
    cv2.putText(frame, "Get " + str(Score), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
    cv2.putText(frame, "FPS " + str(fps), (w - 300, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 255), 3)
    cv2.imshow('GrabStrawberry', frame)
    cv2.moveWindow('GrabStrawberry', 400, 300)
    key = cv2.waitKey(1)
    if key == ord('a'):
        cv2.imwrite('GrabStrawberry.jpg', frame)
    if key == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()