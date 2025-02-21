#!/usr/bin/env python
import rospy
import cv2
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from grounding_sam_ros.client import SamDetector
from grounding_sam_ros.srv import VitDetection
from std_srvs.srv import SetBool, SetBoolResponse

class GroundingSAMNode:
    def __init__(self):
        rospy.init_node('grounding_sam_node')
        
        # 初始化参数
        self.current_prompt = rospy.get_param("~default_prompt", "keyboard. mouse. cellphone. laptop. water cup. ")
        self.bridge = CvBridge()
        self.latest_image = None
        
        # 初始化检测器
        self.detector = SamDetector()
        
        # 图像订阅
        rospy.Subscriber("/camera/color/image_raw", Image, self.image_callback)
        
        # 标注图像发布
        self.annotated_pub = rospy.Publisher("~annotated", Image, queue_size=1)
        
        # Prompt更新服务
        rospy.Service("~update_prompt", SetBool, self.prompt_callback)
        
        # 10Hz检测定时器
        rospy.Timer(rospy.Duration(1/15), self.detection_timer_callback)
        
        rospy.loginfo("Node initialization complete")

    def image_callback(self, msg):
        """图像接收回调"""
        try:
            self.latest_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            rospy.logerr(f"Image conversion failed: {str(e)}")

    def prompt_callback(self, req):
        """Prompt更新服务"""
        self.current_prompt = req.data
        rospy.loginfo(f"Updated prompt to: {self.current_prompt}")
        return SetBoolResponse(success=True, message="Prompt updated")

    def detection_timer_callback(self, event):
        """定时检测回调"""
        if self.latest_image is None:
            return

        try:
            # 执行检测
            annotated, _, _, labels, _ = self.detector.detect(
                self.latest_image, 
                self.current_prompt
            )
            
            # 发布标注结果
            self.annotated_pub.publish(
                self.bridge.cv2_to_imgmsg(annotated, "bgr8")
            )
            
            # 打印检测结果
            if len(labels) > 0:
                rospy.loginfo(f"Detected: {', '.join(labels)}")
                
        except rospy.ServiceException as e:
            rospy.logerr(f"Detection failed: {str(e)}")
        except Exception as e:
            rospy.logerr(f"Processing error: {str(e)}")

if __name__ == '__main__':
    try:
        node = GroundingSAMNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

# #!/usr/bin/env python
# import rospy
# from sensor_msgs.msg import Image
# from std_msgs.msg import String, Float32MultiArray
# from grounding_sam_ros.srv import VitDetection, VitDetectionResponse, VitDetectionRequest
# from std_srvs.srv import SetBool, SetBoolResponse

# class GroundingSAMNode:
#     def __init__(self):
#         # 初始化节点
#         rospy.init_node('grounding_sam_node', anonymous=True)
        
#         # 存储prompt的成员变量
#         self.current_prompt = "computer. keyboard. mouse. cellphone."  # 默认提示词
        
#         # 订阅RGB图像话题
#         self.color_image = None
#         rospy.Subscriber("/camera/color/image_raw", Image, self.image_callback)
        
#         # 发布标注后的图像
#         self.annotated_pub = rospy.Publisher("/annotated_frames", Image, queue_size=10)
        
#         # 创建prompt更新服务
#         rospy.Service('update_prompt', SetBool, self.handle_prompt_update)
        
#         # 创建VitDetection服务客户端
#         rospy.wait_for_service('vit_detection')
#         self.vit_client = rospy.ServiceProxy('vit_detection', VitDetection)
        
#         rospy.loginfo("Node initialized")

#     def image_callback(self, msg):
#         """图像话题回调函数"""
#         self.color_image = msg
#         # rospy.logdebug("Received new color image")

#     def handle_prompt_update(self, req):
#         """Prompt更新服务回调"""
#         # 这里使用SetBool服务的data字段携带新prompt（需转为字符串）
#         new_prompt = str(req.data)
#         self.current_prompt = new_prompt
#         rospy.loginfo(f"Prompt updated to: {new_prompt}")
#         return SetBoolResponse(success=True, message="Prompt updated")

#     def process_results(self, response):
#         """专用结果处理函数（可扩展升级）"""
#         # 当前仅发布标注后的图像
#         if response.annotated_frame is not None:
#             self.annotated_pub.publish(response.annotated_frame)
#             rospy.logdebug("Published annotated frame")

#     def run_detection(self):
#         """主检测逻辑"""
#         rate = rospy.Rate(10)  # 10Hz
#         while not rospy.is_shutdown():
#             if self.color_image is not None:
#                 try:
#                     # 构建服务请求
#                     req = VitDetectionRequest()
#                     req.color_image = self.color_image
#                     req.prompt = self.current_prompt
                    
#                     # 调用服务
#                     response = self.vit_client(req)
                    
#                     # 处理返回数据
#                     self.process_results(response)
                    
#                 except rospy.ServiceException as e:
#                     rospy.logerr(f"Service call failed: {e}")
#             rate.sleep()

# if __name__ == '__main__':
#     try:
#         node = GroundingSAMNode()
#         node.run_detection()
#     except rospy.ROSInterruptException:
#         pass