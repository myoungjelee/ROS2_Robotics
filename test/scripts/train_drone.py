# 주석: 경로 관리를 위해 Pathlib 사용 (시니어 사수의 추천 방식)
from pathlib import Path
from ultralytics import YOLO

def main():
    # 1. 파일 경로 설정
    # 주석: 현재 파일(train_drone.py)의 위치를 기준으로 상위 폴더의 datasets을 찾는다
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent  # test 폴더
    data_yaml = project_root / "datasets" / "data.yaml"

    # 2. 모델 로드 (nano)
    model = YOLO("yolov8n.pt").to("cuda") 

    # 3. 학습 시작
    # 주석: 하이엔드 GPU 사양에 맞춰 파라미터를 최적화한다
    model.train(
        data=str(data_yaml),    # Path 객체를 문자열로 변환
        epochs=15,              # 학습 횟수
        imgsz=640,              # 이미지 크기 (표준)
        batch=32,               # 4070 Ti Super는 32도 가능
        device=0,               # GPU 번호 0번 사용
        workers=8,              # 데이터 로딩용 CPU 코어 개수
        project=str(project_root / "train" / "drone_ai"),     # 결과 저장 폴더 이름
        name="v1_baseline",     # 실험 버전 이름
        exist_ok=True           # 폴더 중복 허용
    )

    print(f"학습 완료! 결과물 위치: {project_root}/train/drone_ai/v1_baseline")

if __name__ == "__main__":
    main()