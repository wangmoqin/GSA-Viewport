import os
import json
import imageio
from tqdm import tqdm
import numpy as np
import cv2
import torch
import math
import time
time_list = []
def p2e_gpu(persp_img, fov_deg, u_deg, v_deg, pano_h, pano_w, device='cuda'):
    fov_h, fov_v = map(math.radians, fov_deg)
    u_rad = math.radians(u_deg)
    v_rad = math.radians(v_deg)

    h, w = persp_img.shape

    cx = w / 2
    cy = h / 2
    fx = cx / math.tan(fov_h / 2)
    fy = cy / math.tan(fov_v / 2)

    ys, xs = torch.meshgrid(
        torch.arange(pano_h, device=device, dtype=torch.float32),
        torch.arange(pano_w, device=device, dtype=torch.float32),
        indexing='ij'
    )

    theta = 2 * math.pi * (xs / pano_w - 0.5)
    phi = math.pi * (ys / pano_h - 0.5)

    dx = torch.cos(phi) * torch.sin(theta)
    dy = torch.sin(phi)
    dz = torch.cos(phi) * torch.cos(theta)

    cos_v = math.cos(v_rad)
    sin_v = math.sin(v_rad)
    cos_u = math.cos(u_rad)
    sin_u = math.sin(u_rad)

    rot_x = torch.tensor([
        [1, 0, 0],
        [0, cos_v, -sin_v],
        [0, sin_v, cos_v]
    ], device=device, dtype=torch.float32)

    rot_y = torch.tensor([
        [cos_u, 0, sin_u],
        [0, 1, 0],
        [-sin_u, 0, cos_u]
    ], device=device, dtype=torch.float32)

    rot = rot_y @ rot_x

    vec = torch.stack([dx, dy, dz], dim=0).reshape(3, -1)
    vec_rot = rot.T @ vec

    z = vec_rot[2, :]
    valid_mask = z > 0

    x_p = fx * (vec_rot[0, :] / z) + cx
    y_p = fy * (vec_rot[1, :] / z) + cy

    x_p_valid = x_p[valid_mask]
    y_p_valid = y_p[valid_mask]

    x_norm = (x_p_valid / (w - 1)) * 2 - 1
    y_norm = (y_p_valid / (h - 1)) * 2 - 1

    grid = torch.stack([x_norm, y_norm], dim=1).unsqueeze(0).unsqueeze(0)

    persp_img_t = persp_img.unsqueeze(0).unsqueeze(0).to(device)

    sampled_vals = torch.nn.functional.grid_sample(
        persp_img_t, grid, mode='bilinear', padding_mode='zeros', align_corners=True
    ).view(-1)

    out = torch.zeros(pano_h * pano_w, device=device, dtype=torch.float32)
    indices = torch.nonzero(valid_mask, as_tuple=False).squeeze(1)
    out[indices] = sampled_vals

    return out.reshape(pano_h, pano_w)

# 参数
pano_w, pano_h = 3840, 2048
fov_deg = (140, 120)
saliency_root = "outputs/fused"
annotation_json = "outputs/sod_annotations_fused.json"
output_dir = "outputs/fused_back"
os.makedirs(output_dir, exist_ok=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

with open(annotation_json, 'r') as f:
    coco_data = json.load(f)

images = coco_data['images']
annotations = coco_data['annotations']

canvas_dict = {}

for image in tqdm(images, desc=" Back-projecting"):
    start_time = time.time()  #  记录每张图开始处理时间

    image_id = image['id']
    file_path = image['file_name']
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    pano_group = os.path.dirname(file_path)  # 作为合成单位（文件夹名）

    # 找到对应 annotation
    anns = [ann for ann in annotations if ann['image_id'] == image_id]

    persp_img_path = os.path.join(saliency_root, file_path)
    if not os.path.exists(persp_img_path):
        print(f"透视图显著性图不存在: {persp_img_path}")
        continue

    sal = imageio.imread(persp_img_path)
    if sal.ndim == 3:
        sal = cv2.cvtColor(sal, cv2.COLOR_RGB2GRAY)
    sal = sal.astype(np.float32) / 255.0
    sal_t = torch.from_numpy(sal).to(device)

    if len(anns) == 0:
        print(f"图像 {base_name} 没有对应annotation")
        continue

    for ann in anns:
        bbox = ann['bbox']
        cx = bbox[0]
        cy = bbox[1]
        theta = bbox[2]
        phi = bbox[3]

        pano_sal = p2e_gpu(
            persp_img=sal_t,
            fov_deg=fov_deg,
            u_deg=theta,
            v_deg=phi,
            pano_h=pano_h,
            pano_w=pano_w,
            device=device
        )

        if pano_group not in canvas_dict:
            canvas_dict[pano_group] = {
                "map": torch.zeros((pano_h, pano_w), dtype=torch.float32, device=device),
                "count": torch.zeros((pano_h, pano_w), dtype=torch.float32, device=device)
            }

        canvas_dict[pano_group]["map"] += pano_sal
        canvas_dict[pano_group]["count"] += (pano_sal > 0).float()

    time_list.append(time.time() - start_time)

# 保存结果
for pano_group, data in canvas_dict.items():
    count = data["count"]
    count[count == 0] = 1
    result = data["map"] / count
    result = (result * 255).clamp(0, 255).byte().cpu().numpy()
    save_path = os.path.join(output_dir, pano_group + ".png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    imageio.imwrite(save_path, result)
    print(f"保存反投影结果: {save_path}")

# 显示平均耗时
if len(time_list) > 0:
    avg_time = np.mean(time_list)
    total_time = np.sum(time_list)
    print(f"\n平均每张图反投影时间: {avg_time * 1000:.2f} ms")
    print(f"总处理时间: {total_time:.2f} 秒")
else:
    print(" 没有图像完成反投影，无法统计时间。")
