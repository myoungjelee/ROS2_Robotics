# 주석: 코랩 전용 패치는 삭제하고 표준 라이브러리 사용
import cv2
import requests
from ultralytics import YOLO

# 1. 모델 로드 (명제님의 RTX 4070 Ti Super 활용)
# 주석: 명시적으로 cuda를 설정하여 GPU를 사용한다
model = YOLO("yolov8n.pt").to("cuda") 

# 2. 테스트용 이미지 다운로드
# 주석: !wget 대신 requests를 사용하여 로컬에 저장한다
url = "https://ultralytics.com/images/zidane.jpg"
image_path = "test_image.jpg"

response = requests.get(url)
with open(image_path, 'wb') as f:
    f.write(response.content)

# 3. Object Detection 수행
results = model.predict(source=image_path, save=True)

# 4. 결과값 출력 (터미널)
class_names = model.names
for box in results[0].boxes:
    x_min, y_min, x_max, y_max = box.xyxy[0]
    confidence = box.conf[0]
    class_id = int(box.cls[0])
    class_name = class_names[class_id]

    print(confidence)
    print(f"Box coordinates: ({x_min}, {y_min}, {x_max}, {y_max})")
    print(f"Confidence score: {confidence:.2f}")
    print(f"Class ID: {class_id}")
    print(f"Class Name: {class_name}\n")


# 5. 결과 시각화 (로컬 윈도우 창)
# 주석: cv2_imshow 대신 표준 cv2.imshow를 사용한다
results_plot = results[0].plot()
cv2.imshow("YOLOv8 Detection Result", results_plot)

# 주석: 창을 바로 닫지 않고 키 입력을 기다린다
cv2.waitKey(0)
cv2.destroyAllWindows()