#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import torch
from unidepth.models import UniDepthV2
from unidepth.utils import colorize


class UniDepthROS2Node(Node):
    def __init__(self):
        super().__init__('unidepth_node')
        
        # Declare parameters
        self.declare_parameter('model_name', 'unidepth-v2-vits14')
        self.declare_parameter('input_topic', '/image_raw')
        self.declare_parameter('depth_topic', '/depth/image_raw')
        self.declare_parameter('depth_colorized_topic', '/depth/image_colorized')
        self.declare_parameter('vmin', 0.1)
        self.declare_parameter('vmax', 10.0)
        
        # Get parameters
        model_name = self.get_parameter('model_name').value
        input_topic = self.get_parameter('input_topic').value
        depth_topic = self.get_parameter('depth_topic').value
        depth_colorized_topic = self.get_parameter('depth_colorized_topic').value
        self.vmin = self.get_parameter('vmin').value
        self.vmax = self.get_parameter('vmax').value
        
        # Initialize CV Bridge
        self.bridge = CvBridge()
        
        # Initialize the model
        self.get_logger().info("Loading UniDepth model...")
        self.model = UniDepthV2.from_pretrained(f"lpiccinelli/{model_name}")
        
        # Move model to GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device).eval()
        self.get_logger().info(f"Model loaded on {self.device}")
        
        # Create subscriber
        self.image_sub = self.create_subscription(
            Image,
            input_topic,
            self.image_callback,
            10
        )
        
        # Create publishers
        self.depth_pub = self.create_publisher(Image, depth_topic, 10)
        self.depth_colorized_pub = self.create_publisher(Image, depth_colorized_topic, 10)
        
        self.get_logger().info(f"Subscribed to: {input_topic}")
        self.get_logger().info(f"Publishing depth to: {depth_topic}")
        self.get_logger().info(f"Publishing colorized depth to: {depth_colorized_topic}")
        
    def image_callback(self, msg):
        try:
            # Convert ROS Image message to OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
            
            # Convert to torch tensor and rearrange dimensions
            rgb_torch = torch.from_numpy(cv_image).permute(2, 0, 1)  # HWC to CHW
            
            # Predict depth
            with torch.no_grad():
                predictions = self.model.infer(rgb_torch)
            
            # Extract depth prediction
            depth_pred = predictions["depth"].squeeze().cpu().numpy()
            
            # Publish raw depth image (32FC1)
            depth_msg = self.bridge.cv2_to_imgmsg(depth_pred, encoding='32FC1')
            depth_msg.header = msg.header
            self.depth_pub.publish(depth_msg)
            
            # Colorize depth for visualization
            depth_colored = colorize(depth_pred, vmin=self.vmin, vmax=self.vmax, cmap="magma_r")
            
            # Publish colorized depth image
            depth_colorized_msg = self.bridge.cv2_to_imgmsg(depth_colored, encoding='rgb8')
            depth_colorized_msg.header = msg.header
            self.depth_colorized_pub.publish(depth_colorized_msg)
            
            self.get_logger().debug('Depth estimation published successfully')
            
        except Exception as e:
            self.get_logger().error(f'Error processing image: {str(e)}')


def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = UniDepthROS2Node()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f'Error: {e}')
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
