import os
import cv2
import numpy as np
import time

method_dirs = ['method1', 'method2', 'method3']
root_dir = 'result'  # 根目录
output_dir = os.path.join(root_dir, 'fused')
os.makedirs(output_dir, exist_ok=True)

# 假设method1目录里所有子文件夹名称列表即所有需要处理的文件夹
base_method_dir = os.path.join(root_dir, method_dirs[0])
subfolders = [f for f in os.listdir(base_method_dir) if os.path.isdir(os.path.join(base_method_dir, f))]

# 计时开始
start_time = time.time()

total_images = 0

for folder_name in subfolders:
    print(f"处理子文件夹: {folder_name}")

    # 创建对应输出子文件夹
    save_folder = os.path.join(output_dir, folder_name)
    os.makedirs(save_folder, exist_ok=True)

    # 获取该子文件夹下所有图片（以method1为基准）
    folder_path_method1 = os.path.join(base_method_dir, folder_name)
    image_files = [f for f in os.listdir(folder_path_method1) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    for img_name in image_files:
        total_images += 1
        fused = None
        missing = False

        for method in method_dirs:
            img_path = os.path.join(root_dir, method, folder_name, img_name)
            if not os.path.exists(img_path):
                print(f' 缺失文件: {img_path}')
                missing = True
                break

            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f' 加载失败: {img_path}')
                missing = True
                break

            img = img.astype(np.float32)
            if fused is None:
                fused = img
            else:
                fused += img

        if missing:
            continue

        # 归一化融合图
        fused = np.clip(fused, 0, 255 * len(method_dirs))
        fused = (fused / fused.max()) * 255
        fused = fused.astype(np.uint8)

        save_path = os.path.join(save_folder, img_name)
        cv2.imwrite(save_path, fused)
        print(f" Saved: {save_path}")

end_time = time.time()
total_time = end_time - start_time
avg_time_per_image = total_time / total_images if total_images > 0 else 0
fps = total_images / total_time if total_time > 0 else 0

print(f"\nTest Done for {total_images} images")
print(f"Total time       : {total_time:.2f} seconds")
print(f"Average per image: {avg_time_per_image*1000:.2f} ms")
print(f"FPS (images/sec) : {fps:.2f}")
