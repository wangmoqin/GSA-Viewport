import torch
import os
import imageio
import numpy as np
import json
from py360convert import e2p
from dataset3 import CocoDataset
from model2 import TransformerWithGCN
from tqdm import tqdm
from sklearn.cluster import DBSCAN
import time
time_list = []
model_path = r"D:\pythondaima\detr-main\6 3,512,adj阈值0.3，学习率5e-4，45轮.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
image_root = r"dataset\output_images"
annotation_file = "dataset/instances_yolo_output.json"

transformer_args = {
    "d_model": 512,
    "nhead": 16,
    "num_encoder_layers": 6,
    "num_decoder_layers": 3,
    "dim_feedforward": 512,
    "dropout": 0.2,
    "activation": "gelu",
    "normalize_before": False,
    "return_intermediate_dec": False
}
gcn_in_features = 512
gcn_out_features = 512
drop_rate = 0.2

# ====== 加载模型 ======
model = TransformerWithGCN(transformer_args, gcn_in_features, gcn_out_features, drop_rate).to(device)
checkpoint = torch.load(model_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
print(f" Loaded model from {model_path}, trained for {checkpoint['epoch']} epochs")

# ====== 加载数据集 ======
# dataset = CocoDataset(annotation_file, os.path.join(image_root, "detect"))
dataset = CocoDataset(annotation_file, os.path.join(image_root))
# ====== 创建输出根目录 ======
output_root = "outputs/sod_views"
os.makedirs(output_root, exist_ok=True)

# ====== COCO annotation dict ======
coco_dict = {
    "images": [],
    "annotations": [],
    "categories": [{"id": 1, "name": "salient", "supercategory": "object"}]
}
image_id = 1
annotation_id = 1

def compute_fov_center(proposal_centers, panoW=3840, panoH=2048):
    theta_list, phi_list = [], []
    for cx, cy in proposal_centers:
        theta = (cx / panoW) * 360 - 180
        phi = 90 - (cy / panoH) * 180
        theta_list.append(theta)
        phi_list.append(phi)

    theta_rad = np.radians(theta_list)
    phi_rad = np.radians(phi_list)

    x = np.cos(phi_rad) * np.cos(theta_rad)
    y = np.cos(phi_rad) * np.sin(theta_rad)
    z = np.sin(phi_rad)

    x_mean = np.mean(x)
    y_mean = np.mean(y)
    z_mean = np.mean(z)
    norm = np.linalg.norm([x_mean, y_mean, z_mean]) + 1e-8
    x_mean /= norm
    y_mean /= norm
    z_mean /= norm

    phi_center = np.arcsin(z_mean)
    theta_center = np.arctan2(y_mean, x_mean)

    return np.degrees(theta_center), np.degrees(phi_center)

for idx in tqdm(range(len(dataset)), desc=" Processing images"):
    start_time = time.time()

    proposals, proposal_features, adj, padding, filename = dataset[idx]
    proposals = proposals.unsqueeze(0).to(device)
    proposal_features = proposal_features.unsqueeze(0).to(device)
    adj = adj.unsqueeze(0).to(device)
    padding = padding.unsqueeze(0).to(device)

    with torch.no_grad():
        logits, _ = model(proposals, proposal_features, adj, padding)
        probs = torch.sigmoid(logits).squeeze(0).squeeze(-1).cpu()

    padding_mask = ~padding.squeeze(0).cpu()
    proposals_cpu = proposals.squeeze(0).cpu()
    conf_threshold = 0.55
    salient_indices = (probs > conf_threshold) & padding_mask

    pano_path = filename
    pano_img = imageio.imread(pano_path)
    base_name = os.path.splitext(os.path.basename(filename))[0]
    image_output_dir = os.path.join(output_root, base_name)
    os.makedirs(image_output_dir, exist_ok=True)

    if salient_indices.sum() == 0:
        print(f" No salient object detected in {base_name}")
        time_list.append(time.time() - start_time)
        continue

    salient_coords = []
    for i, is_salient in enumerate(salient_indices):
        if not is_salient:
            continue
        x, y, w, h = proposals_cpu[i].tolist()
        cx, cy = x + w / 2, y + h / 2
        salient_coords.append((cx, cy))

    if len(salient_coords) == 0:
        time_list.append(time.time() - start_time)
        continue

    coords_np = np.array(salient_coords)
    if len(coords_np) == 3:
        groups = [coords_np]
    else:
        clustering = DBSCAN(eps=1024, min_samples=1).fit(coords_np)
        groups = [coords_np[clustering.labels_ == label] for label in set(clustering.labels_)]

    for gidx, group in enumerate(groups):
        subgroups = [group[i:i + 3] for i in range(0, len(group), 3)]

        for sgidx, subgroup in enumerate(subgroups):
            theta, phi = compute_fov_center(subgroup.tolist())
            persp = e2p(
                pano_img,
                fov_deg=(140, 120),
                u_deg=theta,
                v_deg=phi,
                out_hw=(2048, 3840)
            )

            save_name = f"{base_name}_cluster_{gidx}_part_{sgidx}.png"
            save_path = os.path.join(image_output_dir, save_name)
            imageio.imwrite(save_path, persp)

            mean_cx, mean_cy = np.mean(subgroup, axis=0).tolist()

            coco_dict["images"].append({
                "id": image_id,
                "file_name": os.path.join(base_name, save_name).replace("\\", "/"),
                "width": 1024,
                "height": 512
            })

            coco_dict["annotations"].append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": 1,
                "bbox": [mean_cx, mean_cy, theta, phi],
                "area": 0,
                "iscrowd": 0
            })

            image_id += 1
            annotation_id += 1

    time_list.append(time.time() - start_time)

# ====== 保存 COCO annotation ======
output_json_path = os.path.join(output_root, "sod_annotations_fused.json")
with open(output_json_path, 'w') as f:
    json.dump(coco_dict, f, indent=2)
print(f"\nCOCO annotation saved to: {output_json_path}")

# ====== 显示平均处理时间 ======
avg_time = np.mean(time_list)
print(f"\nAverage processing time per image: {avg_time * 1000:.2f} ms")
print("\n🎉 Finished!")