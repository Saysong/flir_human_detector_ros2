import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray, Detection2D, BoundingBox2D, ObjectHypothesisWithPose
from vision_msgs.msg import Pose2D as VisionPose2D
from cv_bridge import CvBridge
import cv2
import numpy as np

class FlirHumanDetectorNode(Node):
    def __init__(self):
        super().__init__('flir_human_detector_node')

        self.declare_parameter('thermal_image_topic', '/thermal/image_raw')
        self.declare_parameter('detection_topic', '/human_detections')
        self.declare_parameter('threshold_value', 120)
        self.declare_parameter('min_contour_area', 10)
        self.declare_parameter('max_contour_area', 500)

        thermal_image_topic = self.get_parameter('thermal_image_topic').get_parameter_value().string_value
        self.detection_topic = self.get_parameter('detection_topic').get_parameter_value().string_value
        self.threshold_value = self.get_parameter('threshold_value').get_parameter_value().integer_value
        self.min_contour_area = self.get_parameter('min_contour_area').get_parameter_value().integer_value
        self.max_contour_area = self.get_parameter('max_contour_area').get_parameter_value().integer_value

        self.subscription = self.create_subscription(
            Image,
            thermal_image_topic,
            self.image_callback,
            10
        )

        self.publisher_ = self.create_publisher(Detection2DArray, self.detection_topic, 10)
        self.image_pub = self.create_publisher(Image, self.detection_topic + '/image', 10)
        self.bridge = CvBridge()

        self.get_logger().info(f'Thermal Human Detector Node started, subscribing to {thermal_image_topic}')

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            _, thresh = cv2.threshold(blurred, self.threshold_value, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            detections = Detection2DArray()
            detections.header = msg.header
            display = cv_image.copy()

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if self.min_contour_area <= area <= self.max_contour_area:
                    x, y, w, h = cv2.boundingRect(cnt)

                    bbox = BoundingBox2D()
                    bbox.center.position.x = float(x + w / 2)
                    bbox.center.position.y = float(y + h / 2)
                    bbox.size_x = float(w)
                    bbox.size_y = float(h)

                    detection = Detection2D()
                    detection.header = msg.header
                    detection.bbox = bbox

                    hypothesis = ObjectHypothesisWithPose()
                    hypothesis.hypothesis.class_id = "human"
                    hypothesis.hypothesis.score = 0.9  # static confidence
                    detection.results.append(hypothesis)

                    detections.detections.append(detection)
                    cv2.rectangle(display, (x, y), (x + w, y + h), (0, 0, 255), 2)

            self.publisher_.publish(detections)
            image_msg = self.bridge.cv2_to_imgmsg(display, encoding='bgr8')
            image_msg.header = msg.header
            self.image_pub.publish(image_msg)

            self.get_logger().info(f'Detected {len(detections.detections)} humans')

        except Exception as e:
            self.get_logger().error(f'Processing error: {e}')

def main(args=None):
    rclpy.init(args=args)
    node = FlirHumanDetectorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
