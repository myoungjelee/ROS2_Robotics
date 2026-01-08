#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
from transformers import CLIPProcessor, CLIPModel
import torch
from PIL import Image as PILImage


class VLMAnalysisNode(Node):
    def __init__(self):
        super().__init__('vlm_analysis_node')

        self.subscription = self.create_subscription(
            Image,
            'input_image',
            self.image_callback,
            10
        )
        self.publisher = self.create_publisher(String, 'vlm_output', 10)
        self.bridge = CvBridge()
        self.get_logger().info("VLM Analysis Node Started.")

        # CLIP 모델과 프로세서를 로드
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.model.eval()

        # ------------------------------------------------------------
        # (기존) 매번 cuda 여부를 확인하고 model.to("cuda") 호출
        # if torch.cuda.is_available():
        #     self.model.to("cuda")
        #
        # (변경) 디바이스를 "한 번만" 결정해서 멤버로 들고가자
        # 왜? 콜백마다 체크/분기하면 쓸데없는 반복이고 코드도 더러워짐
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # 비교할 텍스트 리스트 (학습용이면 하드코딩 OK)
        self.text_inputs = ["a photo of a cat", "a photo of a dog", "a photo of a car"]

    def image_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        pil_image = PILImage.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))

        inputs = self.processor(
            text=self.text_inputs,
            images=pil_image,
            return_tensors="pt",
            padding=True
        )

        # ------------------------------------------------------------
        # (기존) 매 콜백마다 cuda 체크 + dict로 올림
        # if torch.cuda.is_available():
        #     inputs = {k: v.to("cuda") for k, v in inputs.items()}
        #
        # (변경) 이미 self.device를 정했으니 그걸로만 올리자
        # 왜? cuda 여부 체크를 여기서 할 이유가 없음 (한 번 정했으면 끝)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        outputs = self.model(**inputs)
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1)

        result_text = "Similarity scores:\n"
        for text, prob in zip(self.text_inputs, probs[0].tolist()):
            result_text += f"'{text}': {prob:.4f}\n"

        self.get_logger().info(result_text)

        output_msg = String()
        output_msg.data = result_text
        self.publisher.publish(output_msg)


def main(args=None):
    rclpy.init(args=args)
    node = VLMAnalysisNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
