import cv2
import torch
import os
from torch.utils.data import Dataset
from PIL import Image
from pycocotools.coco import COCO
from torchvision import transforms
import numpy as np
import scipy.sparse as sp
from torchvision.ops import roi_align
from torchvision.models import resnet50
import torch.nn as nn
from PIL import ImageOps

MAX_NUM_PROPOSALS = 20
def resize_bbox(bbox, orig_width, orig_height, new_width, new_height):
    if bbox == [-1, -1, -1, -1]:
        return bbox
    x_min, y_min, box_width, box_height = bbox
    scale_x = new_width / orig_width
    scale_y = new_height / orig_height

    x_min = int(x_min * scale_x)
    y_min = int(y_min * scale_y)

    box_width = int(box_width * scale_x)
    box_height = int(box_height * scale_y)

    x_min = max(0, min(x_min, new_width - 1))
    y_min = max(0, min(y_min, new_height - 1))
    box_width = max(1, min(box_width, new_width - x_min))
    box_height = max(1, min(box_height, new_height - y_min))

    return [x_min, y_min, box_width, box_height]

def normalize(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -0.5).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    return r_mat_inv.dot(mx).dot(r_mat_inv)


def compute_iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    inter_x1 = max(x1, x2)
    inter_y1 = max(y1, y2)
    inter_x2 = min(x1 + w1, x2 + w2)
    inter_y2 = min(y1 + h1, y2 + h2)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0
def generate_adj(proposals, max_num_proposals):
    num_proposals = proposals.size(0)
    iou_threshold = 0.1
    edges_unordered = []
    for i in range(num_proposals):
        for j in range(i + 1, num_proposals):
            iou = compute_iou(proposals[i].tolist(), proposals[j].tolist())
            if iou > iou_threshold:
                edges_unordered.append([i, j])
                edges_unordered.append([j, i])
    edges_unordered = np.array(edges_unordered)
    if edges_unordered.ndim == 1:
        edges_unordered = edges_unordered.reshape(-1, 2)
    adj = sp.coo_matrix(
        (np.ones(len(edges_unordered)),
         (np.array(edges_unordered)[:, 0], np.array(edges_unordered)[:, 1])),
        shape=(num_proposals, num_proposals),
        dtype=np.float32
    )
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = normalize(adj + sp.eye(adj.shape[0]))
    adj = torch.tensor(adj.toarray(), dtype=torch.float32)

    if num_proposals < max_num_proposals:
        adj_padded = torch.zeros(max_num_proposals, max_num_proposals, dtype=torch.float32)
        adj_padded[:num_proposals, :num_proposals] = adj
        adj = adj_padded
    elif num_proposals > max_num_proposals:
        adj = adj[:max_num_proposals, :max_num_proposals]

    return adj


class CocoDataset(Dataset):
    def __init__(self, annotation_file, image_root, max_size=(2048,3840), transform=None, return_masks=False):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.coco = COCO(annotation_file)
        self.image_root = image_root
        self.image_ids = self.coco.getImgIds()
        self.max_size = max_size
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])
        self.feature_extractor = resnet50(pretrained=True)
        self.feature_extractor = torch.nn.Sequential(*list(self.feature_extractor.children())[:-4])
        self.feature_extractor = self.feature_extractor.to(device)
        self.feature_extractor.eval()
        # self.conv1 = nn.Conv2d(2048, 512, kernel_size=1).to(device)

    def load_heatmap(self, img_path):
        img_id = os.path.splitext(os.path.basename(img_path))[0]
        heatmap_filename = f"{img_id}_gt.npy"
        heatmap_path = os.path.join(os.path.dirname(img_path), heatmap_filename)

        if not os.path.exists(heatmap_path):
            raise FileNotFoundError(f"Heatmap file not found: {heatmap_path}")

        heatmap = np.load(heatmap_path)
        if heatmap.ndim == 3 and heatmap.shape[0] == 1:
            heatmap = heatmap[0]

        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Image file not found or unreadable: {img_path}")
        h, w = img.shape[:2]

        heatmap_resized = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_LINEAR)

        return heatmap_resized

    def extract_proposal_features(self, image, proposals):

        if image.dim() == 3:
            image = image.unsqueeze(0)

        with torch.no_grad():
            image_features = self.feature_extractor(image)
        proposals = proposals.clone()
        proposals[:, 2] = proposals[:, 0] + proposals[:, 2]  # x2 = x1 + w
        proposals[:, 3] = proposals[:, 1] + proposals[:, 3]  # y2 = y1 + h
        proposal_features = roi_align(
            image_features,
            [proposals],
            output_size=(7, 7),
            spatial_scale=image_features.size(-1) / image.size(-1)
        )

        proposal_features = torch.mean(proposal_features, dim=(-2, -1))  # [N, C]

        return proposal_features

    def resize_with_aspect_ratio(self, image, max_dim):
        orig_width, orig_height = image.size
        scale = min(max_dim[0] / orig_height, max_dim[1] / orig_width)
        new_width = int(orig_width * scale)
        new_height = int(orig_height * scale)
        resized_image = image.resize((new_width, new_height), Image.BILINEAR)
        return resized_image, scale

    def pad_image(self, image):
        orig_width, orig_height = image.size
        target_height, target_width = self.max_size

        pad_top = (target_height - orig_height) // 2
        pad_bottom = target_height - orig_height - pad_top
        pad_left = (target_width - orig_width) // 2
        pad_right = target_width - orig_width - pad_left
        padded_image = ImageOps.expand(image, (pad_left, pad_top, pad_right, pad_bottom), fill=(0, 0, 0))
        return padded_image, pad_left, pad_top


    def is_salient(self, heatmap, proposals, top_k=5):

        salient_scores = []

        for bbox in proposals:
            x_min, y_min, width, height = map(int, bbox)

            x_max = min(x_min + width, heatmap.shape[1])
            y_max = min(y_min + height, heatmap.shape[0])

            mean_value = np.mean(heatmap[y_min:y_max, x_min:x_max])

            salient_scores.append((mean_value, len(salient_scores)))

        salient_scores.sort(reverse=True, key=lambda x: x[0])
        topk_indices = {idx for _, idx in salient_scores[:top_k]}

        labels = torch.tensor([1 if i in topk_indices else 0 for i in range(len(proposals))], dtype=torch.long,
                              device=proposals.device)

        return labels

    def adjust_bbox(self, bbox, scale, pad_left, pad_top):
        if bbox == [-1, -1, -1, -1]:
            return bbox
        x, y, w, h = bbox
        x *= scale
        y *= scale
        w *= scale
        h *= scale
        return [x + pad_left, y + pad_top, w, h]

    def __getitem__(self, idx):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        img_id = self.image_ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        orig_width, orig_height = img_info['width'], img_info['height']
        img_path = os.path.join(self.image_root, img_info['file_name'].replace('/', os.sep))

        bbox = torch.tensor([ann['bbox'] for ann in anns], dtype=torch.float32, device=device)
        num_proposals = bbox.size(0)
        if num_proposals < MAX_NUM_PROPOSALS:
            padding = torch.full((MAX_NUM_PROPOSALS - num_proposals, 4), -1, device=bbox.device)  # 用零填充
            bbox = torch.cat([bbox, padding], dim=0)
        elif num_proposals > MAX_NUM_PROPOSALS:
            bbox = bbox[:MAX_NUM_PROPOSALS]

        padding_mask = (bbox[:, 0] == -1)


        image = Image.open(img_path).convert("RGB")
        image, scale = self.resize_with_aspect_ratio(image, self.max_size)
        image, pad_x, pad_y = self.pad_image(image)
        image = self.transform(image).to(device)
        proposal = [self.adjust_bbox(b, scale, pad_x, pad_y) for b in bbox]
        proposal = torch.tensor(proposal, dtype=torch.float32).to(device)

        proposal_features = self.extract_proposal_features(image, proposal).to(device)

        adj = generate_adj(proposal, MAX_NUM_PROPOSALS).to(device)

        return proposal, proposal_features, adj, padding_mask, img_path

    def __len__(self):
        return len(self.image_ids)
