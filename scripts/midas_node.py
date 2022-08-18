#!/usr/bin/env python3

import cv2
import numpy as np
import rospy
import torch
from sensor_msgs.msg import CompressedImage


class MIDAS:

    def __init__(self):

        rospy.init_node("midas_node", anonymous=True)
        self.pub = rospy.Publisher("/midas/depth/image_raw/compressed", CompressedImage, queue_size=1, tcp_nodelay=True)
        # rospy.Subscriber("/camera/color/image_raw/compressed", CompressedImage, self.compressed_image_callback)
        rospy.Subscriber("equirectangular/image_raw/compressed", CompressedImage, self.compressed_image_callback, tcp_nodelay=True)
        self.input_image = np.empty(0)
        self.compressed_image = CompressedImage()
        self.compressed_image.format = "jpeg"

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.midas = torch.hub.load("intel-isl/MiDaS", "DPT_Hybrid")
        self.midas.to(self.device).eval()
        self.transforms = torch.hub.load("intel-isl/MiDaS", "transforms").dpt_transform

    def compressed_image_callback(self, data: CompressedImage) -> None:

        self.input_image = cv2.imdecode(np.frombuffer(data.data, np.uint8), cv2.IMREAD_COLOR)  # type: ignore
        self.compressed_image.header = data.header

    def estimate_depth(self, _) -> None:

        if self.input_image.shape[0] == 0:
            return

        with torch.no_grad():
            input = self.transforms(self.input_image).to(self.device)
            output = self.midas(input)
            result = torch.nn.functional.interpolate(  # type: ignore
                output.unsqueeze(1),
                size=self.input_image.shape[:2],
                mode="bilinear",
                align_corners=False).squeeze().cpu().numpy()
        result = (result - np.min(result)) / (np.max(result) - np.min(result)) * 255
        self.compressed_image.data = cv2.imencode(".jpg", result)[1].squeeze().tolist()  # type: ignore
        self.pub.publish(self.compressed_image)

    def __call__(self) -> None:

        rospy.Timer(rospy.Duration(nsecs=500000), self.estimate_depth)
        rospy.spin()


def main() -> None:

    MIDAS()()


if __name__ == "__main__":
    main()
