#!/usr/bin/env python3
# 위의 shebang은 이 스크립트가 Python3 인터프리터로 실행되도록 지정합니다.

import rclpy                     # ROS2 Python 클라이언트 라이브러리를 가져옵니다.
from rclpy.node import Node      # ROS2 노드 클래스를 가져와서 사용자 정의 노드를 생성할 수 있게 합니다.
from sensor_msgs.msg import Image  # ROS2의 이미지 메시지 타입을 가져옵니다.
from cv_bridge import CvBridge   # OpenCV 이미지와 ROS 이미지 메시지 간의 변환을 위해 CvBridge를 임포트합니다.
import cv2                       # OpenCV 라이브러리를 가져와서 카메라 캡처 및 영상 처리를 수행합니다.

# CameraPublisherNode 클래스는 ROS2 노드로, 외부 카메라(예: Logitech C920)에서 영상을 캡처하고 ROS2 토픽에 발행합니다.
class CameraPublisherNode(Node):
    def __init__(self):
        # 부모 클래스(Node)의 생성자를 호출하면서 노드 이름을 'camera_publisher_node'로 지정합니다.
        super().__init__('camera_publisher_node')
        
        # ROS2 토픽 'input_image'에 sensor_msgs/Image 메시지를 발행할 퍼블리셔를 생성합니다.
        self.publisher_ = self.create_publisher(Image, 'input_image', 10)
        
        # CvBridge 객체를 생성하여 OpenCV 이미지와 ROS 이미지 메시지 간 변환에 사용합니다.
        self.bridge = CvBridge()
        
        # 타이머 주기를 0.1초(10Hz)로 설정하고, timer_callback 함수를 주기적으로 호출합니다.
        timer_period = 0.1  # 0.1초마다 콜백 호출
        self.timer = self.create_timer(timer_period, self.timer_callback)
        
        # 외부 카메라(예: Logitech C920)를 사용하기 위해 ls /dev/video* 명령어를 입력하여 내가 연결한 카메라가 몇번 인덱스인지 확인합니다.
        # 외부 카메라를 연결하지 않고 내장 카메라를 연결하고 싶다면 0번으로 넣어주면 됩니다. 
        self.cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))  # MJPG로 강제
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        # 카메라가 정상적으로 열렸는지 확인합니다.
        if not self.cap.isOpened():
            self.get_logger().error("외부 카메라(인덱스 0)를 열 수 없습니다. 연결 상태와 인덱스를 확인하세요.")
        else:
            self.get_logger().info("외부 카메라가 정상적으로 열렸습니다.")
        
        # OpenCV를 사용하여 영상 창("Camera View")을 생성합니다.
        # WINDOW_NORMAL 옵션을 사용하면 창 크기를 자유롭게 조절할 수 있습니다.
        cv2.namedWindow("Camera View", cv2.WINDOW_NORMAL)

    # timer_callback 함수는 타이머가 호출될 때마다 실행되어 카메라 프레임을 캡처, 발행, 그리고 화면에 표시합니다.
    def timer_callback(self):
        # 카메라에서 프레임을 읽어옵니다.
        ret, frame = self.cap.read()
        # 프레임 캡처가 성공했는지 확인합니다.
        if ret:
            # 읽어온 프레임을 "Camera View" 창에 표시합니다.
            cv2.imshow("Camera View", frame)
            # OpenCV 창이 정상적으로 업데이트되도록 1밀리초 대기합니다.
            cv2.waitKey(1)
            
            # CvBridge를 사용하여 OpenCV의 프레임을 ROS2 이미지 메시지로 변환합니다.
            # encoding='bgr8'은 OpenCV의 기본 색상 순서(BGR)를 지정합니다.
            image_msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
            # 변환된 이미지 메시지를 'input_image' 토픽에 발행합니다.
            self.publisher_.publish(image_msg)
            # 로그 메시지를 통해 프레임 발행 사실을 출력합니다.
            self.get_logger().info("카메라 프레임 발행")
        else:
            # 프레임 캡처에 실패했을 경우 에러 로그를 출력합니다.
            self.get_logger().error("프레임 캡처 실패")

    # 노드 종료 시 호출되는 함수로, 카메라 리소스를 해제하고 모든 OpenCV 창을 닫습니다.
    def destroy_node(self):
        # 카메라가 열려 있다면 리소스를 해제합니다.
        if self.cap.isOpened():
            self.cap.release()
        # 모든 OpenCV 창을 닫습니다.
        cv2.destroyAllWindows()
        # 부모 클래스의 destroy_node()를 호출하여 추가 리소스를 정리합니다.
        super().destroy_node()

# 메인 함수: ROS2 환경을 초기화하고 노드를 실행합니다.
def main(args=None):
    # ROS2 클라이언트를 초기화합니다.
    rclpy.init(args=args)
    # CameraPublisherNode의 인스턴스를 생성합니다.
    node = CameraPublisherNode()
    try:
        # 노드를 실행하며, 메시지 콜백 등을 계속해서 처리합니다.
        rclpy.spin(node)
    except KeyboardInterrupt:
        # 사용자가 Ctrl+C로 인터럽트를 발생시키면 예외 처리합니다.
        pass
    finally:
        # 노드를 종료하고 ROS2 클라이언트를 정리합니다.
        node.destroy_node()
        rclpy.shutdown()

# 이 스크립트가 메인 모듈로 실행될 경우 main() 함수를 호출합니다.
if __name__ == '__main__':
    main()