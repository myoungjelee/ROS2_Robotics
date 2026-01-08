#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

import torch
# from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer


class TextGeneratorNode(Node):
    def __init__(self):
        super().__init__('text_generator_node')

        # 입력 토픽 "input_text" 구독, 출력 토픽 "output_text" 발행
        self.subscription = self.create_subscription(
            String,
            'input_text',
            self.listener_callback,
            10,
        )
        self.publisher = self.create_publisher(String, 'output_text', 10)
        self.get_logger().info("Text Generator Node Started.")

        # GPT-2 모델과 토크나이저 로드
        # self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        # self.model = GPT2LMHeadModel.from_pretrained("gpt2")
        self.tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        self.model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0",torch_dtype=torch.float16,)
        self.model.eval()

        # GPU가 있다면 GPU 사용
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def listener_callback(self, msg):
        input_text = msg.data
        self.get_logger().info("Received input: " + input_text)

        # 입력 텍스트를 토큰화
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt").to(self.device)

        # 텍스트 생성 (최대 50 토큰)
        output_ids = self.model.generate(
            input_ids,
            # max_length=50,    
            max_new_tokens=50,  # 새로 생성할 토큰 수 (max_length는 입력포함 토큰 수)
            num_return_sequences=1,
        )

        # generated_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        generated_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True,)
        self.get_logger().info("Generated text: " + generated_text)

        # 생성된 텍스트를 발행
        output_msg = String()
        output_msg.data = generated_text
        self.publisher.publish(output_msg)


def main(args=None):
    rclpy.init(args=args)
    node = TextGeneratorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
