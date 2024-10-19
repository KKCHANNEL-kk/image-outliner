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
    stroke_size = stroke_size - 1
    mat = np.ones(input_mat.shape)
    check_size = stroke_size + 1.0
    mat[input_mat > check_size] = 0  # 距离大于描边宽度的部分设为0
    border = (input_mat > stroke_size) & (input_mat <= check_size)
    mat[border] = 1.0 - (input_mat[border] - stroke_size)  # 描边宽度内的部分线性调整
    return mat


class ImageOutliner:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "stroke_size": ("INT", {"default": 1, "min": 1, "max": 100}),
                "colors": ("STRING", {"default": "#000000"}),
                "padding": ("INT", {"default": 0, "min": 0, "max": 100})
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "outline_image"

    CATEGORY = "api/image"

    def outline_image(self, image: torch.Tensor, threshold: float, stroke_size: int, colors: str, padding: int):
        # 将torch.Tensor转换为PIL图像
        i = 255. * image.cpu().numpy()
        
        # 去掉批次维度
        if i.shape[0] == 1:
            i = i.squeeze(0)
        
        img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
        
        # 将颜色字符串转换为RGB元组
        colors = tuple(int(colors.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))

        # 将PIL图像转换为NumPy数组
        img_array = np.array(img)
        h, w, _ = img_array.shape
        alpha = img_array[:,:,3]
        rgb = img_array[:,:,0:3]

        # 在RGB图像和Alpha通道周围添加填充
        bigger_img = cv2.copyMakeBorder(rgb, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=(0, 0, 0,0))
        alpha = cv2.copyMakeBorder(alpha, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=0)
        bigger_img = cv2.merge((bigger_img, alpha))
        h, w, _ = bigger_img.shape

        # 去除阴影，将Alpha通道二值化
        _, alpha_without_shadow = cv2.threshold(alpha, int(threshold * 255), 255, cv2.THRESH_BINARY)
        alpha_without_shadow = 255 - alpha_without_shadow

        # 计算距离变换，获取每个像素到最近背景像素的距离
        dist = cv2.distanceTransform(alpha_without_shadow, cv2.DIST_L2, cv2.DIST_MASK_3)

        # 对距离变换矩阵进行高斯模糊处理
        dist = cv2.GaussianBlur(dist, (5, 5), 0)

        # 根据描边宽度调整距离变换矩阵
        stroked = change_matrix(dist, stroke_size)  # 根据描边宽度调整距离变换矩阵
        stroke_alpha = (stroked * 255).astype(np.uint8)  # 将调整后的矩阵转换为Alpha通道

        # 创建描边颜色通道
        stroke_b = np.full((h, w), colors[2], np.uint8)
        stroke_g = np.full((h, w), colors[1], np.uint8)
        stroke_r = np.full((h, w), colors[0], np.uint8)

        # 合并描边颜色和Alpha通道
        stroke = cv2.merge((stroke_b, stroke_g, stroke_r, stroke_alpha))

        # 将OpenCV图像转换为PIL图像
        stroke = Image.fromarray(stroke)
        # DEBUG：导出 stroke
        stroke.save("stroke.png")
        bigger_img = Image.fromarray(bigger_img)

        # 将描边图像与原图像进行Alpha合成
        result = Image.alpha_composite(stroke, bigger_img)

        # 将PIL图像转换回torch.Tensor
        result_array = np.array(result).astype(np.float32) / 255.0
        result_img = torch.from_numpy(result_array)[None,]

        return [result_img]

# A dictionary that contains all nodes you want to export with their names.
NODE_CLASS_MAPPINGS = {
    "ImageOutliner": ImageOutliner
}
