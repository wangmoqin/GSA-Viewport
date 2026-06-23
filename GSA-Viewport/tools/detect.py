import os
import time  # ✅ 用于时间统计
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms
import pathlib
pathlib.PosixPath = pathlib.WindowsPath
import model.CII as module_arch

# 配置
input_dir = r'D:\pythondaima\后处理显著性图生成\CII-main\data\sod_views'
output_dir = r'D:\pythondaima\后处理显著性图生成\CII-main\method3'
checkpoint_path = r'D:\pythondaima\后处理显著性图生成\CII-main\saved\models\cii.pth'
input_size = 352
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 模型加载
print("加载模型...")
model = module_arch.CII(
    base='resnet50',
    convert=[64, 256, 512, 1024, 2048],
    center=64,
    topdown=[[True, True, True, True, False],
             [True, True, True, True, False]],
    score=64
)
checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint['state_dict'])
model.to(device)
model.eval()
print("模型加载完毕。")

# 图像预处理
preprocess = transforms.Compose([
    transforms.Resize((input_size, input_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# 收集图片
img_suffix = ('.jpg', '.jpeg', '.png')
image_files = []
for root, dirs, files in os.walk(input_dir):
    for file in files:
        if file.lower().endswith(img_suffix):
            full_path = os.path.join(root, file)
            rel_path = os.path.relpath(full_path, input_dir)
            image_files.append(rel_path)

print(f"共发现 {len(image_files)} 张图片。")

# 统计时间
time_list = []
total_start_time = time.time()  # ⏱️ 总开始时间

for rel_path in image_files:
    try:
        abs_path = os.path.join(input_dir, rel_path)
        img = Image.open(abs_path).convert('RGB')
        w, h = img.size

        input_tensor = preprocess(img).unsqueeze(0).to(device)

        start_time = time.time()  # ⏱️ 单张开始时间
        with torch.no_grad():
            output = model(input_tensor)
            if isinstance(output, (list, tuple)):
                output = output[0]
            output = F.interpolate(output, size=(h, w), mode='bilinear', align_corners=False)
            output = torch.sigmoid(output).cpu().squeeze().numpy()
        time_list.append(time.time() - start_time)  # ⏱️ 单张处理时间

        # 归一化与保存
        output = (output - output.min()) / (output.max() - output.min() + 1e-8)
        output = (output * 255).astype(np.uint8)

        save_path = os.path.join(output_dir, rel_path)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        Image.fromarray(output).convert('L').save(save_path)
        print(f"保存结果: {save_path}")

    except Exception as e:
        print(f"处理失败 {rel_path}: {e}")

# 总时间统计
total_end_time = time.time()
total_time = total_end_time - total_start_time
avg_time = np.mean(time_list)
fps = 1.0 / avg_time if avg_time > 0 else 0

# 打印统计结果
print("\n推理统计结果：")
print(f"总处理时间：{total_time:.2f} 秒")
print(f"平均每张图处理时间：{avg_time * 1000:.2f} 毫秒")
print(f"平均处理速度（FPS）：{fps:.2f} 张/秒")
