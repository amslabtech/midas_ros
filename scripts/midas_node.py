#!/usr/bin/env python3

import os

import cv2
import numpy as np
import rospy
import torch
from sensor_msgs.msg import CompressedImage


class MiDaS:
    _hz: float
    _pixel_divisor: float
    _model_type: str
    _use_amp: bool
    _pub: rospy.Publisher
    _sub: rospy.Subscriber
    _input_image: np.ndarray
    _compressed_image: CompressedImage
    _device: torch.device
    _midas: torch.nn.Module
    _transforms: torch.nn.Module

    def __init__(self) -> None:

        rospy.init_node("midas_node", anonymous=True)

        self._hz = rospy.get_param("~hz", 15.0)  # type: ignore
        self._pixel_divisor = rospy.get_param("~pixel_divisor", 2.5)  # type: ignore
        self._model_type = rospy.get_param("~model_type", "DPT_Hybrid")  # type: ignore
        self._use_amp = rospy.get_param("~use_amp", True)  # type: ignore

        self._pub = rospy.Publisher(
            "/midas/depth/image_raw/compressed",
            CompressedImage, queue_size=1, tcp_nodelay=True)
        self._sub = rospy.Subscriber(
            "/camera/color/image_raw/compressed",
            CompressedImage, self._compressed_image_callback, queue_size=1, tcp_nodelay=True)
        self._input_image = np.empty(0)
        self._images = np.empty(0)
        self._compressed_image = CompressedImage()
        self._compressed_image.format = "jpeg"

        torch.backends.cudnn.benchmark = True  # type: ignore
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        midas_path = os.path.join(torch.hub.get_dir(), "intel-isl_MiDaS_master")
        self._midas = (
            torch.hub.load(midas_path, self._model_type, source="local")
            if os.path.exists(midas_path)
            else torch.hub.load("intel-isl/MiDaS", self._model_type)
        ).to(self._device).eval()
        self._transforms = (
            torch.hub.load(midas_path, "transforms", source="local")
            if os.path.exists(midas_path)
            else torch.hub.load("intel-isl/MiDaS", "transforms")
        ).dpt_transform

    def _compressed_image_callback(self, data: CompressedImage) -> None:

        decoded_image = cv2.imdecode(np.frombuffer(data.data, np.uint8), cv2.IMREAD_COLOR)  # type: ignore
        self._input_image = cv2.resize(
            decoded_image,
            (int(decoded_image.shape[1]/self._pixel_divisor), int(decoded_image.shape[0]/self._pixel_divisor)))
        self._compressed_image.header = data.header

    def _estimate_depth(self, _) -> None:

        if self._input_image.shape[0] == 0:
            return

        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=self._use_amp):  # type: ignore
                input = self._transforms(self._input_image).to(self._device)
                output = self._midas(input)
            result = torch.nn.functional.interpolate(  # type: ignore
                output.unsqueeze(1),
                size=self._input_image.shape[:2],
                mode="bilinear",
                align_corners=False).float().squeeze().cpu().numpy()

        result = np.uint8(cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX))  # type: ignore
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
