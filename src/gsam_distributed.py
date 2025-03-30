import rospy
from groundingdino.util.inference import load_model, load_image, predict, annotate
from groundingdino.util.inference import Model
import cv2
import torch

from grounding_sam_ros.srv import VitDetection, VitDetectionResponse
from cv_bridge import CvBridge
from std_msgs.msg import MultiArrayDimension

# SAM
from segment_anything import SamPredictor, sam_model_registry
import numpy as np
import supervision as sv

import os
from sensor_msgs.msg import Image
from grounding_sam_ros.msg import AnnotationInfo, MaskInfo

LINKS = {
    "SAM-H": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
    "SAM-L": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
    "SAM-B": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
    "DINO": "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth",
}
MODELS = {
    "SAM-H": "vit_h",
    "SAM-L": "vit_l",
    "SAM-B": "vit_b",
}

class DetectSegmentation:
    def __init__(self, model_path, config, sam_checkpoint, sam_model, box_threshold=0.4, text_threshold=0.3):
        torch.cuda.empty_cache()
        if torch.cuda.is_available():    
            self.device = torch.device("cuda")
        else:
            print("No GPU available")
            exit()

        # rgb图像订阅
        self.image_sub = rospy.Subscriber("color_to_process", Image, self.det_seg, queue_size=1)
        
        # 图像标注信息(labels,boxes) 和 掩码信息(labels,scores,segmasks) 发布者
        self.annotation_info_pub = rospy.Publisher("annotation_info", AnnotationInfo, queue_size=1)
        self.mask_info_pub = rospy.Publisher("mask_info", MaskInfo, queue_size=1)

        self.cv_bridge = CvBridge()

        rospy.loginfo("Loading groundingdino and sam models...")

        # Building GroundingDINO inference model
        if not os.path.exists(model_path):
            rospy.loginfo("Downloading DINO model...")
            if not os.path.exists("weights"):
                os.makedirs("weights")
            os.system("wget {} -O {}".format(LINKS["DINO"], model_path))
        self.grounding_dino_model = Model(model_config_path=config, model_checkpoint_path=model_path)
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold

        # Building SAM inference model
        if not os.path.exists(sam_checkpoint):
            rospy.loginfo("Downloading SAM model...")
            if not os.path.exists("weights"):
                os.makedirs("weights")
            os.system("wget {} -O {}".format(LINKS[sam_model], sam_checkpoint))
        self.sam_predictor = SamPredictor(sam_model_registry[MODELS[sam_model]](checkpoint=sam_checkpoint).to(self.device))

        rospy.set_param("/yolo_evsam_ros_init", True)
        rospy.loginfo("groundingdino and sam models are loaded")

        # # ros service
        # self.cv_bridge = CvBridge()
        # rospy.Service("vit_detection", VitDetection, self.callback)
        # rospy.loginfo("vit_detection service has started")

    def det_seg(self, image_msg):
        start_time = rospy.Time.now()

        # 将参数服务器里的det_seg_processing设置成True，防止定时器在前一张图像的检测分割还没完成时发布新图像
        rospy.set_param("det_seg_processing", True)

        # 提取图像时间戳，将图像消息转化为np数组形式
        time_stamp = image_msg.header.stamp
        image = self.cv_bridge.imgmsg_to_cv2(image_msg, "bgr8")

        # 从参数服务器获取检测类型（检测提示词）
        class_list = rospy.get_param("detection_prompt")
        # print(class_list)

        # 目标检测和语义分割
        text = ". ".join(class_list) + "."  # str
        detections, labels, masks = self.detect(image, text)

        if len(detections.confidence) == 0:
            rospy.loginfo("no detection")
            rospy.set_param("det_seg_processing", False)
            return

        # 测量检测分割的时间
        end_time = rospy.Time.now()
        seg_time = (end_time - start_time).to_sec()*1000
        rospy.loginfo(f"detect+segment time: {seg_time:.1f} ms")

        masks = masks.astype(np.uint8)  # True -> 1, False -> 0
        # masks = (masks.astype(np.uint8)) * 255  # True -> 255(白色), False -> 0(黑色)
        
        # 将 N 张 H×W 的掩码图转换为 单个 H×W×N 的多通道 Image 消息
        masks_stacked = np.stack(masks, axis=-1)  # 变成 (H, W, N)

        class_id = detections.class_id
        boxes = detections.xyxy
        scores = detections.confidence
        masks = detections.mask
        

        # 发布图像标注信息
        annotation_info = AnnotationInfo()
        annotation_info.header.stamp = time_stamp
        annotation_info.class_id = class_id
        annotation_info.labels = labels
        annotation_info.boxes.layout.dim = [MultiArrayDimension(label="boxes", size=boxes.shape[0], stride=4)]
        annotation_info.boxes.data = boxes.flatten().tolist()
        self.annotation_info_pub.publish(annotation_info)

        # 发布掩码信息
        mask_info = MaskInfo()
        mask_info.header.stamp = time_stamp
        mask_info.labels = labels
        mask_info.scores = scores.tolist()
        mask_info.segmasks = self.cv_bridge.cv2_to_imgmsg(masks_stacked, encoding="passthrough")  # 8 位无符号整数，N 通道
        self.mask_info_pub.publish(mask_info)

        # # We release the gpu memory
        # torch.cuda.empty_cache()

        # 将参数服务器里的det_seg_processing设置成False，允许定时器发布新图像
        rospy.set_param("det_seg_processing", False)

    def segment(self, image: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
        self.sam_predictor.set_image(image)
        result_masks = []
        for box in xyxy:
            masks, scores, logits = self.sam_predictor.predict(
                box=box,
                multimask_output=True
            )
            index = np.argmax(scores)
            result_masks.append(masks[index])
        return np.array(result_masks)

    def detect(self, image, text):
        # GroundingDINO Model
        # detect objects
        detections, labels = self.grounding_dino_model.predict_with_caption(
                    image=image,
                    caption=text,
                    box_threshold=self.box_threshold,
                    text_threshold=self.text_threshold
                )
        
        # >>> yutian >>>
        phrases = labels
        if text[-1]=='.':
            text += ' '
        classes = text.split('. ')
        class_id = self.grounding_dino_model.phrases2classes(phrases=phrases, classes=classes)
        detections.class_id = class_id
        # <<< yutian <<<

        # Segment Anything Model
        # convert detections to masks
        masks = self.segment(
            image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
            xyxy=detections.xyxy
        )

        # # annotate image with detections
        # # mask_annotator = sv.MaskAnnotator()
        # box_annotator = sv.BoxAnnotator()
        # label_annotator = sv.LabelAnnotator()
        
        # # annotated_image = mask_annotator.annotate(scene=image.copy(), detections=detections)
        # annotated_image = box_annotator.annotate(scene=image.copy(), detections=detections)
        # annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)

        return detections, labels, masks
    
if __name__ == '__main__':
    rospy.init_node('grounding_sam_ros')

    # get arguments from the ros parameter server
    model_path = rospy.get_param('~model_path')
    config = rospy.get_param('~config')
    sam_checkpoint = rospy.get_param('~sam_checkpoint')
    sam_model = rospy.get_param('~sam_model')
    box_threshold = rospy.get_param('~box_threshold')
    text_threshold = rospy.get_param('~text_threshold')

    # start the server
    DetectSegmentation(model_path, config, sam_checkpoint, sam_model, box_threshold, text_threshold)
    rospy.spin()