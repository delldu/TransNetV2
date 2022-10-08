"""Video Scene Package."""  # coding=utf-8
#
# /************************************************************************************
# ***
# ***    Copyright Dell 2022(18588220928@163.com) All Rights Reserved.
# ***
# ***    File Author: Dell, 2022年 10月 08日 星期六 21:25:27 CST
# ***
# ************************************************************************************/
#

__version__ = "1.0.0"

import os
from tqdm import tqdm
import torch

import redos
import todos

from PIL import Image
import numpy as np

from . import scene

import pdb


DETECT_IMAGE_HEIGHT = 27
DETECT_IMAGE_WIDTH = 48


def get_model():
    """Create model."""

    model_path = "models/video_scene.pth"
    cdir = os.path.dirname(__file__)
    checkpoint = model_path if cdir == "" else cdir + "/" + model_path

    model = scene.TransNetV2()
    todos.model.load(model, checkpoint)
    device = todos.model.get_device()
    model = model.to(device)
    model.eval()

    print(f"Running on {device} ...")
    model = torch.jit.script(model)

    todos.data.mkdir("output")
    if not os.path.exists("output/video_scene.torch"):
        model.save("output/video_scene.torch")

    return model, device


def video_predict(input_file, output_file):
    # load video
    video = redos.video.Reader(input_file)
    # video = redos.video.Sequence(input_file)
    print(video)

    if video.n_frames < 1:
        print(f"Read video {input_file} error.")
        return False

    # Create directory to store result
    output_dir = output_file[0 : output_file.rfind("/")]
    if output_dir != None and output_dir != "":
        todos.data.mkdir(output_dir)

    # load model
    model, device = get_model()

    frame_list = []
    print(f"  process {input_file}, save to {output_file} ...")

    print("Start decode video ...")
    decode_progress_bar = tqdm(total=video.n_frames)

    def reading_video_frames(no, data):
        decode_progress_bar.update(1)
        # print(f"frame: {no} -- {data.shape}"),
        # data -- np.frombuffer(buffer, np.uint8).reshape([self.height, self.width, 4])

        image = Image.fromarray(data).resize((DETECT_IMAGE_WIDTH, DETECT_IMAGE_HEIGHT)).convert("RGB")
        tensor = torch.from_numpy(np.array(image)).unsqueeze(0)  # 1x3x27x48, range [0, 255]

        frame_list.append(tensor)

    video.forward(callback=reading_video_frames)
    del decode_progress_bar

    no_padded_frames_start = 25
    no_padded_frames_end = 25 + 50 - (len(frame_list) % 50 if len(frame_list) % 50 != 0 else 50)  # 25 - 74
    # len(frames) -- 14541 ==> len(frames) % 50 --- 41
    # ==> no_padded_frames_end -- 34

    start_frame = frame_list[0]
    end_frame = frame_list[-1]
    padded_list = [start_frame] * no_padded_frames_start + frame_list + [end_frame] * no_padded_frames_end

    print("Start scene detection ... ")
    predict_list = []
    detect_progress_bar = tqdm(total=len(padded_list))
    for index in range(0, len(padded_list), 50):
        detect_progress_bar.update(50)
        input_tensor = torch.cat(padded_list[index : index + 100], dim=0)

        # single_frame_pred, all_frame_pred = model(input_tensor)
        single_frame_pred = todos.model.forward(model, device, input_tensor)

        single_frame_pred = torch.sigmoid(single_frame_pred).cpu()  # [1, 100, 1]
        predict_list.append(single_frame_pred[:, 25:75, :])

    predict_list = predict_list[0 : len(frame_list)]
    prediction_tensor = torch.cat(predict_list, dim=1)

    # torch.where(prediction_tensor > 0.5)[1].size() -- torch.Size([110])
    # index -- torch.where(prediction_tensor > 0.5)[1].tolist()

    start = 0
    sbd_list = []
    for index in torch.where(prediction_tensor > 0.5)[1].tolist():
        if index > start + 5:  # One scene at least has 5 frames
            sbd_list.append((start, index))
            start = index + 1  # next
    if start < len(frame_list) - 5:  # One scene at least has 5 frames
        sbd_list.append((start, len(frame_list) - 1))

    with open(output_file, "w") as f:
        for (s, e) in sbd_list:
            f.write(f"{s} {e}\n")

    todos.model.reset_device()

    return True
