"""
opencv 라이브러리를 통해
웹캠 이미지를 YOLO로 보내주는 기본 코드


(아래 주석은 문제가 발생하기 전에는 읽지 않으셔도 돼요.)

ImportError 발생시
ultralytics, supervision 및 python-opencv 설치해야 함.
> pip install ultralytics
> pip install opencv-python -i https://pypi.tuna.tsinghua.edu.cn/simple

ffmpeg 관련 오류 발생시?
ffmpeg를 설치하고 환경변수에 추가해야 함
윈도우용 다운로드 링크는 https://www.gyan.dev/ffmpeg/builds/ 에서
"ffmpeg-git-full.7z" 클릭하여 다운로드 후 임의의 폴더에 압축을 풀고
bin 폴더를 환경변수에 추가하면 됨(사용자/시스템 환경변수 무관)

리눅스의 경우에는 https://ffmpeg.org/download.html#build-linux 에서 동일하게 다운로드하면 됨
압축 푼 후 ~/.bashrc의 PATH 관련 라인이나 마지막 라인에 PATH 환경변수에 해당 경로 추가.
만약 zsh 등 다른 터미널 실행시에는 ~/.zshrc 등에 환경변수 추가하면 됨.
예시 : `$PATH=$PATH:~/util/ffmpeg/bin`

윈도우에서 opencv 버전 관련 오류 발생하면(4.9 이상으로 업데이트 필요)
pip uninstall opencv-python
pip uninstall opencv-contrib-python
pip install opencv-python -i https://pypi.tuna.tsinghua.edu.cn/simple
"""

from ultralytics import YOLO
import cv2


camera = cv2.VideoCapture(0)  # 디바이스 선택
model = YOLO("yolov8n.pt")  # 추론모델 선택

while True:
    img = camera.read()[1]
    results = model(img, show=True, save=True)
    # ↑ YOLO가 띄우는 창이지만 내부적으로는 opencv와 동일합니다.
    # 매번 results라는 객체를 새로 생성하므로, stream=True 옵션을 줄 필요가 없습니다.
    # 대신 과거에 생성된 추론정보는 모두 지워지고 있습니다.

    if cv2.waitKey(1) == ord('q'):  # 실행 도중 q를 입력하면 캡쳐 종료
        break

camera.release()
cv2.destroyAllWindows()
