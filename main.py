from PIL import Image
import numpy as np
import cv2
import torch


def change_matrix(input_mat: np.ndarray, stroke_size: int) -> np.ndarray:
    """
    根据描边宽度调整距离变换矩阵。

    参数:
    input_mat (np.ndarray): 输入的距离变换矩阵
    stroke_size (int): 描边宽度

    返回:
    np.ndarray: 调整后的矩阵
    """
    mat = np.ones(input_mat.shape)
    check_size = stroke_size + 1.0
    mat[input_mat > check_size] = 0  # 距离大于描边宽度的部分设为0
    border = (input_mat > stroke_size) & (input_mat <= check_size)
    mat[border] = 1.0 - (input_mat[border] - stroke_size)  # 描边宽度内的部分线性调整
    return mat


def tensor2image(tensor: torch.Tensor) -> Image.Image:
    """
    将 torch.Tensor 转换为 PIL.Image.Image 对象。

    参数:
    tensor (torch.Tensor): 输入的张量，范围假定在 [0.0, 1.0] 之间。

    返回:
    PIL.Image.Image: 转换后的图像。
    """
    # 将张量从 GPU 移动到 CPU，并转换为 NumPy 数组
    numpy_array = tensor.cpu().numpy()

    # 缩放到 [0, 255]，并裁剪到合法范围
    scaled = np.clip(numpy_array * 255.0, 0, 255)

    # 转换为 uint8 类型
    uint8_array = scaled.astype(np.uint8)

    # 创建 PIL 图像
    return Image.fromarray(uint8_array)


class ImageOutliner:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "threshold": (
                    "FLOAT",
                    {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "stroke_size": ("INT", {"default": 1, "min": 1, "max": 100}),
                "colors": ("STRING", {"default": "#000000"}),
                "padding": ("INT", {"default": 0, "min": 0, "max": 100}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "outline_image"

    CATEGORY = "api/image"

    def outline_image(
        self,
        image: torch.Tensor,
        threshold: float,
        stroke_size: int,
        colors: str,
        padding: int,
    ):
        """
        对批量图像进行描边处理，并返回处理后的图像。

        参数:
        images (torch.Tensor): 输入的图像张量，形状为 (batch_size, channels, height, width)
        threshold (float): 阈值，用于二值化Alpha通道
        stroke_size (int): 描边宽度
        colors (str): 描边颜色，十六进制字符串，例如 "#FF0000"
        padding (int): 填充大小

        返回:
        torch.Tensor: 处理后的图像张量，形状为 (batch_size, channels, height, width)
        """
        images = image  # 临时处理，防止工作流崩了
        batch_size = images.shape[0]
        result_images = []

        # 将torch.Tensor转换为NumPy数组并处理每一张图像
        for batch_idx in range(batch_size):
            image = images[batch_idx]

            # 将torch.Tensor转换为PIL图像
            i = 255.0 * image.cpu().numpy()

            # 去掉批次维度
            if i.shape[0] == 1:
                i = i.squeeze(0)

            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))

            # 将颜色字符串转换为RGB元组
            colors_rgb = tuple(
                int(colors.lstrip("#")[j : j + 2], 16) for j in (0, 2, 4)
            )

            # 将PIL图像转换为NumPy数组
            img_array = np.array(img)
            if img_array.shape[2] < 4:
                # 如果没有Alpha通道，添加一个全不透明的Alpha通道
                alpha_channel = np.full(
                    (img_array.shape[0], img_array.shape[1], 1), 255, dtype=np.uint8
                )
                img_array = np.concatenate((img_array, alpha_channel), axis=2)
            h, w, _ = img_array.shape
            alpha = img_array[:, :, 3]
            rgb = img_array[:, :, 0:3]

            # 在RGB图像和Alpha通道周围添加填充
            bigger_rgb = cv2.copyMakeBorder(
                rgb,
                padding,
                padding,
                padding,
                padding,
                cv2.BORDER_CONSTANT,
                value=(0, 0, 0),
            )
            bigger_alpha = cv2.copyMakeBorder(
                alpha, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=0
            )
            bigger_img = cv2.merge((bigger_rgb, bigger_alpha))
            h_new, w_new, _ = bigger_img.shape

            # 去除阴影，将Alpha通道二值化
            _, alpha_without_shadow = cv2.threshold(
                bigger_alpha, int(threshold * 255), 255, cv2.THRESH_BINARY
            )
            alpha_without_shadow = 255 - alpha_without_shadow

            # 计算距离变换，获取每个像素到最近背景像素的距离
            dist = cv2.distanceTransform(
                alpha_without_shadow, cv2.DIST_L2, cv2.DIST_MASK_3
            )

            # 对距离变换矩阵进行高斯模糊处理
            dist = cv2.GaussianBlur(dist, (5, 5), 0)

            # 根据描边宽度调整距离变换矩阵
            stroked = change_matrix(dist, stroke_size)
            stroke_alpha = (stroked * 255).astype(np.uint8)

            # 创建描边颜色通道
            stroke_b = np.full((h_new, w_new), colors_rgb[2], np.uint8)
            stroke_g = np.full((h_new, w_new), colors_rgb[1], np.uint8)
            stroke_r = np.full((h_new, w_new), colors_rgb[0], np.uint8)

            # 合并描边颜色和Alpha通道
            stroke = cv2.merge((stroke_b, stroke_g, stroke_r, stroke_alpha))

            # 将OpenCV图像转换为PIL图像
            stroke_pil = Image.fromarray(stroke)
            bigger_img_pil = Image.fromarray(bigger_img)

            # 将描边图像与原图像进行Alpha合成
            result = Image.alpha_composite(stroke_pil, bigger_img_pil)

            # 将PIL图像转换回torch.Tensor
            result_array = np.array(result).astype(np.float32) / 255.0
            # 调整维度顺序为 (C, H, W)
            result_tensor = torch.from_numpy(result_array)

            result_images.append(result_tensor)

        # 将所有处理后的图像堆叠成一个批次
        result_batch = torch.stack(result_images, dim=0)

        return [result_batch]


# A dictionary that contains all nodes you want to export with their names.
NODE_CLASS_MAPPINGS = {"ImageOutliner": ImageOutliner}
