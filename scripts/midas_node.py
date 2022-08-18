#!/usr/bin/env python3

import numpy as np
import rospy
import torch
from sensor_msgs.msg import CompressedImage


class MiDaS:
    _pub: rospy.Publisher
    _sub: rospy.Subscriber
    _input_image: np.ndarray
    _compressed_image: CompressedImage
    _device: torch.device
    _midas: torch.nn.Module
    _transforms: torch.nn.Module

    def __init__(self) -> None:

        rospy.init_node("midas_node", anonymous=True)

        self._hz = rospy.get_param("~hz", 3)  # type: ignore
        self._pub = rospy.Publisher(
            "/midas/depth/image_raw/compressed",
            CompressedImage, queue_size=1, tcp_nodelay=True)
        self._sub = rospy.Subscriber(
            "/camera/color/image_raw/compressed",
            CompressedImage, self._compressed_image_callback, tcp_nodelay=True)
        self._input_image = np.empty(0)
        self._compressed_image = CompressedImage()
        self._compressed_image.format = "jpeg"

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._midas = torch.hub.load("intel-isl/MiDaS", "DPT_Hybrid")
        self._midas.to(self._device).eval()
        self._transforms = torch.hub.load("intel-isl/MiDaS", "transforms").dpt_transform

    def _compressed_image_callback(self, data: CompressedImage) -> None:

        self._input_image = cv2.imdecode(np.frombuffer(data.data, np.uint8), cv2.IMREAD_COLOR)  # type: ignore
        self._compressed_image.header = data.header

    def _estimate_depth(self, _) -> None:

        if self._input_image.shape[0] == 0:
            return

        with torch.no_grad():
            input = self._transforms(self._input_image).to(self._device)
            output = self._midas(input)
            result = torch.nn.functional.interpolate(  # type: ignore
                output.unsqueeze(1),
                size=self._input_image.shape[:2],
                mode="bilinear",
                align_corners=False).squeeze().cpu().numpy()
        result = (result - np.min(result)) / (np.max(result) - np.min(result)) * 255
        self._compressed_image.data = cv2.imencode(".jpg", result)[1].squeeze().tolist()  # type: ignore
        self._pub.publish(self._compressed_image)

    def __call__(self) -> None:

        duration = int(1.0 / self._hz * 1e6)
        rospy.Timer(rospy.Duration(nsecs=duration), self._estimate_depth)
        rospy.spin()


def main() -> None:

    MiDaS()()


if __name__ == "__main__":
    main()
