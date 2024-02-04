"""
추론결과에서 프레임별로 사람 수 세기 심화버전
result 안의 데이터 실시간으로 다뤄보기
카메라 안에 사람이 있을 때에만
엑셀파일 로그 및 해당이미지 저장하기

% pip install pandas openpyxl ultralytics opencv-python
"""

import os
from time import sleep

from ultralytics import YOLO  # pip install ultralytics
import cv2  # pip install opencv-python
import pandas as pd  # pip install pandas
import datetime as dt

try:
    os.mkdir("log")
except FileExistsError:
    pass

start_time = dt.datetime.now()
df = pd.DataFrame()

camera = cv2.VideoCapture(0)
camera.set(3, 640)  # Width
camera.set(4, 480)  # Height

model = YOLO("yolov8n.pt")
classNames = model.names
# 기본 제공 레이블
# {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train',
#  7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign',
#  12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep',
#  19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella',
#  26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard',
#  32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard',
#  37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork',
#  43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange',
#  50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair',
#  57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv',
#  63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave',
#  69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase',
#  76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}

while True:
    sleep(0.5)
    _, img = camera.read()
    results = model(img, stream=True)
    # ↑ stream=True로 설정하면 <generator> 반환하므로 for문을 돌려야 함

    #
    for r in results:
        boxes = r.boxes  # 박스정보 꺼내서(이게 우리가 분석할 대상)
        # 직접 opencv로 박스도 그리고 레이블도 그려볼 것.
        # 기본창도 YOLO에서 말고, opencv창으로 띄울 예정.

        total_person_no = 0
        for box in boxes:
            # 바운딩박스 위치(좌상, 우하점 좌표) 추출
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # 소숫점 버림

            # 직접 웹캠이미지에 네모 그리기
            cv2.rectangle(img=img,
                          pt1=(x1, y1),
                          pt2=(x2, y2),
                          color=(255, 0, 255),
                          thickness=3)

            # 박스에서 신뢰도 추출
            confidence = box.conf[0]

            # 박스에서 클래스 추출
            cls = int(box.cls[0])

            # 주석 작성
            org = [x1, y1]  # 주석 작성할 좌표
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            color = (255, 0, 0)
            thickness = 2
            cv2.putText(img, classNames[cls] + f" {confidence:.02f}", org, font, fontScale, color, thickness)

            # 로그용 데이터 취합
            agg_data = [{"time": dt.datetime.now(),
                         "count": total_person_no,
                         "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                         "conf": f"{confidence:.02f}"}]

            if cls == 0:  # "person"일 때만 아래 작업 수행
                total_person_no += 1
                df = pd.concat([df, pd.DataFrame(agg_data)], ignore_index=True)

        if total_person_no > 0:
            cv2.imwrite(filename=f".q/{dt.datetime.now():%Y-%m-%d-%H%M%S}_{total_person_no}.jpg",
                        img=img)

    cv2.imshow(winname='Webcam', mat=img)  # opencv 창에 이미지 갱신
    if cv2.waitKey(1) == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
df.to_excel(f"log_{start_time:%Y-%m-%d-%H-%M-%S}.xlsx", index=False)
