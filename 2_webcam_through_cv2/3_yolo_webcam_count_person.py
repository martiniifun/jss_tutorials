"""
보안 관련 간단한 예시코드
실시간 프레임별로 사람 수 세서
"연월일시분초ms : 1" 포맷으로 로그txt 파일 남기기
"""

from ultralytics import YOLO  # pip install ultralytics
import cv2  # pip install opencv-python
import datetime as dt  # 로그파일 작성을 위해 datetime 임포트


# 모델 정의
model = YOLO("yolov8n.pt")

# 카메라 선택 및 설정
camera = cv2.VideoCapture(0)
camera.set(3, 640)  # 출력해상도(3:Width, 4:Height)
camera.set(4, 480)

while True:
    _, img = camera.read()
    results = model(img, show=True, save=True, stream=True,
                    project="infer", name="predict")
    # ↑ stream=True로 설정하면 list 대신 <generator> 반환

    for result in results:
        total_person_no = 0
        boxes = result.boxes
        for box in boxes:  # 각각의 박스 중
            cls = int(box.cls[0])
            if cls == 0:  # 클래스가 0(person)이면
                total_person_no += 1
        print("total person no. : ", total_person_no)
        with open("count_log.txt", "a") as f:
            f.write(f"{dt.datetime.now()} : {total_person_no}\n")

    if cv2.waitKey(30) == ord('q'):  # q 누르면 종료
        break

camera.release()
cv2.destroyAllWindows()
