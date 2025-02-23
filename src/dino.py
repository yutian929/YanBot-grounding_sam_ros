import rospy
from groundingdino.util.inference import load_model, load_image, predict, annotate
from groundingdino.util.inference import Model
import cv2
import torch
import numpy as np

from grounding_sam_ros.srv import VitDetection, VitDetectionResponse
from cv_bridge import CvBridge
from std_msgs.msg import MultiArrayDimension
from groundingdino.util.inference import box_convert
import os

import supervision as sv  # for new API

LINKS = {"DINO": "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth"}

class VitDetectionServer(object):
    def __init__(self, model_path, config, box_threshold=0.4, text_threshold=0.3):
        torch.cuda.empty_cache()
        if torch.cuda.is_available():    
            self.device = torch.device("cuda")
        else:
            print("No GPU available - using CPU")
            self.device = torch.device("cpu") 

        rospy.loginfo("Loading model...")
        if not os.path.exists(model_path):
            rospy.loginfo("Downloading DINO model...")
            if not os.path.exists("weights"):
                os.makedirs("weights")
            os.system("wget {} -O {}".format(LINKS["DINO"], model_path))
        self.grounding_dino_model = Model(model_config_path=config, model_checkpoint_path=model_path)
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        rospy.loginfo("Grounding Dino Model loaded")

        # ros service
        self.cv_bridge = CvBridge()
        rospy.Service("vit_detection", VitDetection, self.callback)
        rospy.loginfo("vit_detection service has started")

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

        # Fake Segment 
        # convert detections to masks
        detections.mask = np.zeros((len(detections.xyxy), image.shape[0], image.shape[1]), dtype=np.uint8)
        for i, box in enumerate(detections.xyxy):
            x1, y1, x2, y2 = box
            detections.mask[i, int(y1):int(y2), int(x1):int(x2)] = 255

        # annotate image with detections
        # mask_annotator = sv.MaskAnnotator()
        box_annotator = sv.BoxAnnotator()
        label_annotator = sv.LabelAnnotator()
        
        # annotated_image = mask_annotator.annotate(scene=image.copy(), detections=detections)
        annotated_image = box_annotator.annotate(scene=image.copy(), detections=detections)
        annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)

        return detections, labels, annotated_image

        

    def callback(self, request):
        img, prompt = request.color_image, request.prompt
        img = self.cv_bridge.imgmsg_to_cv2(img)
        detections, labels, annotated_frame = self.detect(img, prompt)
        boxes = detections.xyxy
        scores = detections.confidence
        masks = detections.mask

        rospy.loginfo("Detected objects: {}".format(labels))
        rospy.loginfo("Detection scores: {}".format(scores))

        
        response = VitDetectionResponse()
        response.labels = labels
        response.class_id = detections.class_id
        response.scores = scores.tolist()
        response.boxes.layout.dim = [MultiArrayDimension(label="boxes", size=boxes.shape[0], stride=4)]
        response.boxes.data = boxes.flatten().tolist()
        response.annotated_frame = self.cv_bridge.cv2_to_imgmsg(annotated_frame)

        # >>> yutian >>>
        try:
            stride = masks.shape[1] * masks.shape[2]
            response.segmasks.layout.dim = [MultiArrayDimension(label="masks", size=masks.shape[0], stride=stride)]
            response.segmasks.data = masks.flatten().tolist()
        except:  # no masks, masks.shape=(0,)
            if masks.size == 0:
                response.segmasks.layout.dim = [MultiArrayDimension(label="masks", size=0, stride=0)]
                response.segmasks.data = []
            else:
                raise ValueError(f"masks.shape is unexpected {masks.shape}")
        # <<< yutian <<<

        # We release the gpu memory
        torch.cuda.empty_cache()
        
        return response

    
if __name__ == '__main__':
    rospy.init_node('grounding_sam_ros')

    # get arguments from the ros parameter server
    model_path = rospy.get_param('~model_path')
    config = rospy.get_param('~config')
    box_threshold = rospy.get_param('~box_threshold')
    text_threshold = rospy.get_param('~text_threshold')

    # start the server
    VitDetectionServer(model_path, config, box_threshold, text_threshold)
    rospy.spin()