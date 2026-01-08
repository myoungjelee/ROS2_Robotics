#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from transformers import pipeline

class SentimentAnalysisNode(Node):
    def __init__(self):
        super().__init__('sentiment_analysis_node')
        # 'sentence_input' 토픽에서 문장을 구독하고, 'sentiment_output' 토픽으로 결과 발행
        self.subscription = self.create_subscription(
            String,
            'sentence_input',
            self.listener_callback,
            10)
        self.publisher = self.create_publisher(String, 'sentiment_output', 10)
        self.get_logger().info("Sentiment Analysis Node Started.")

        # Huggingface의 pipeline을 이용해 감성 분석 파이프라인 생성 (BERT 계열 모델 사용)
        self.classifier = pipeline("sentiment-analysis")
        
    def listener_callback(self, msg):
        sentence = msg.data
        self.get_logger().info("Received sentence: " + sentence)
        # 감성 분석 실행, 결과는 label과 score가 포함된 딕셔너리 리스트로 반환됨
        result = self.classifier(sentence)[0]
        output_text = f"Label: {result['label']}, Score: {result['score']:.4f}"
        self.get_logger().info("Analysis Result: " + output_text)
        # 결과 메시지 생성 후 발행
        output_msg = String()
        output_msg.data = output_text
        self.publisher.publish(output_msg)

def main(args=None):
    rclpy.init(args=args)
    node = SentimentAnalysisNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()