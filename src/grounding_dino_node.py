#!/usr/bin/env python
import rospy
import cv2
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from grounding_sam_ros.client import SamDetector
from grounding_sam_ros.srv import VitDetection
from grounding_sam_ros.srv import UpdatePrompt, UpdatePromptResponse

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
        self.masks_pub = rospy.Publisher("~masks", Image, queue_size=1)

        # Prompt更新服务
        rospy.Service("~update_prompt", UpdatePrompt, self.prompt_callback)
        
        # 定时器
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
        return UpdatePromptResponse(success=True, message=f"Prompt updated to {self.current_prompt}")

    def apply_mask_overlay(self, image, masks):
        """将掩码以半透明固定颜色叠加到图像上"""
        overlay = image.copy()
        
        # 定义固定的颜色列表（RGB格式）
        fixed_colors = [
            [255, 0, 0],    # 红色
            [0, 255, 0],    # 绿色
            [0, 0, 255],    # 蓝色
            [255, 255, 0],  # 黄色
            [255, 0, 255],  # 紫色
            [0, 255, 255],  # 青色
            [128, 0, 0],    # 深红
            [0, 128, 0],    # 深绿
            [0, 0, 128],    # 深蓝
            [128, 128, 0],  # 橄榄色
        ]
        
        for i, mask in enumerate(masks):
            # 使用模运算循环选择颜色
            color = fixed_colors[i % len(fixed_colors)]
            
            # 将二值掩码转换为bool类型
            binary_mask = mask.astype(bool)
            
            # 创建颜色掩码
            color_mask = np.zeros_like(image)
            color_mask[binary_mask] = color
            
            # 使用cv2.addWeighted进行叠加
            alpha = 0.15  # 透明度
            cv2.addWeighted(color_mask, alpha, overlay, 1 - alpha, 0, overlay)
            
            # 绘制轮廓加强显示
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, contours, -1, color, 2)
        
        return overlay

    def detection_timer_callback(self, event):
        """定时检测回调"""
        if self.latest_image is None:
            return

        try:
            # 执行检测
            # annotated_frame, boxes, masks, labels, scores
            annotated, _, masks, labels, _ = self.detector.detect(
                self.latest_image, 
                self.current_prompt
            )
            
            # 发布标注结果
            self.annotated_pub.publish(
                self.bridge.cv2_to_imgmsg(annotated, "bgr8")
            )

            # 生成掩码叠加图像
            if len(masks) > 0:
                mask_overlay = self.apply_mask_overlay(annotated, masks)
                self.masks_pub.publish(
                    self.bridge.cv2_to_imgmsg(mask_overlay, "bgr8")
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