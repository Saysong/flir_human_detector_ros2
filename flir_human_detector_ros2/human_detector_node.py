import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

from flir_lepton_human_detector.human_detector import HumanDetector


class HumanDetectionNode(Node):
    def __init__(self):
        super().__init__('human_detection_node')
        self.declare_parameter('image_topic', '/thermal/image_raw')
        self.declare_parameter('output_topic', '/thermal/detections/image')
        
        self.image_topic = self.get_parameter('image_topic').get_parameter_value().string_value
        self.output_topic = self.get_parameter('output_topic').get_parameter_value().string_value

        self.bridge = CvBridge()
        self.detector = HumanDetector()

        self.sub = self.create_subscription(Image, self.image_topic, self.image_callback, 10)
        self.pub = self.create_publisher(Image, self.output_topic, 10)

        self.get_logger().info(f"Subscribed to {self.image_topic}, publishing to {self.output_topic}")

    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')
        except Exception as e:
            self.get_logger().error(f"CV Bridge error: {e}")
            return

        detections = self.detector.detect(frame)

        for (x, y, w, h) in detections:
            cv2.rectangle(frame, (x, y), (x+w, y+h), 255, 1)

        annotated_msg = self.bridge.cv2_to_imgmsg(frame, encoding='mono8')
        annotated_msg.header = msg.header
        self.pub.publish(annotated_msg)

def main(args=None):
    rclpy.init(args=args)
    node = HumanDetectionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
