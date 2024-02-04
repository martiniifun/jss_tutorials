"""
opencv 라이브러리를 통해
웹캠 이미지를 YOLO로 보내고
향후 분석을 위해 각각의 이미지를 저장하는 예시
"""


from ultralytics import YOLO
import cv2


camera = cv2.VideoCapture(0)
model = YOLO("yolov8n.pt")

while True:
    img = camera.read()[1]  # 사진 찍어서
    results = model(img,  # 분석해.
                    show=True,  # 화면에 띄워주면서
                    save=True,  # 이미지파일로도 저장하고
                    save_txt=True,  # 레이블정보도 txt파일로 저장해
                    project="results",  # 저장할 디렉토리
                    name="predict",  # 각 저장할 서브디렉토리(+자동레이블)
                    )

    if cv2.waitKey(1) == ord('q'):  # 실행 도중에 q를 누르면
        break  # while문(추론) 종료

camera.release()  # 카메라 끄고
cv2.destroyAllWindows()  # 창 닫아.
