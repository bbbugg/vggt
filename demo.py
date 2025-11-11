import json

import numpy as np
import torch
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images

device = "cuda" if torch.cuda.is_available() else "cpu"
# bfloat16 is supported on Ampere GPUs (Compute Capability 8.0+)
dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

# Initialize the model and load the pretrained weights.
# This will automatically download the model weights the first time it's run, which may take a while.
# model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)

model = VGGT()
model.load_state_dict(torch.load("ckpt/model.pt", map_location=device))
model.to(device) # 确保模型及其所有缓冲区都被移动到指定的设备

# Load and preprocess example images (replace with your own image paths)
# image_names = ["path/to/imageA.png", "path/to/imageB.png", "path/to/imageC.png"]
image_names = ["init.png"]
images = load_and_preprocess_images(image_names).to(device)


def save_predictions_to_json(predictions, output_path):
    """
    将 VGGT 模型的预测结果保存为 JSON 格式

    Args:
        predictions: 模型返回的预测结果字典
        output_path: 输出 JSON 文件的路径
    """
    # 创建一个新字典用于存储可序列化的数据
    serializable_preds = {}

    # 处理 predictions 中的每个键值对
    for key, value in predictions.items():
        if isinstance(value, torch.Tensor):
            # 将张量移到 CPU，转换为 numpy 数组，再转换为列表
            serializable_preds[key] = value.detach().cpu().numpy().tolist()
        elif isinstance(value, np.ndarray):
            # 直接将 numpy 数组转换为列表
            serializable_preds[key] = value.tolist()
        elif isinstance(value, list):
            # 如果是列表，检查其中元素的类型
            serializable_list = []
            for item in value:
                if isinstance(item, torch.Tensor):
                    serializable_list.append(item.detach().cpu().numpy().tolist())
                elif isinstance(item, np.ndarray):
                    serializable_list.append(item.tolist())
                else:
                    serializable_list.append(item)
            serializable_preds[key] = serializable_list
        else:
            # 其他基本类型直接保存
            serializable_preds[key] = value

    # 保存为 JSON 文件
    with open(output_path, 'w') as f:
        # indent=2 用于格式化输出，便于阅读
        json.dump(serializable_preds, f, indent=2)

with torch.no_grad():
    with torch.cuda.amp.autocast(dtype=dtype):
        # Predict attributes including cameras, depth maps, and point maps.
        predictions = model(images)
        save_predictions_to_json(predictions, image_names[0].replace(".png", "_predictions.json"))
