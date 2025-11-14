import os
import json
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# === 路径设置 ===
pred_txt_folder = "hongyu_pred/json_results"       # 模型预测标签文件夹
gt_txt_folder = "hongyu_pred/test"    # 正确标签文件夹

# === 类别映射 ===
label_mapping = {
    "Ascarid": 0,
    "Hookworm": 1,
    "Mansoni": 2,
    "Hexacanth larva": 3,
    "Egg sac": 4,
    "Fluke": 5,
    "Whipworm": 6,
    "Coccidian": 7,
    "Starch granule": 8,
    "Muscle fiber": 9,
    "Plant cell": 10
}

# 创建反向映射用于混淆矩阵显示
id_to_label = {v: k for k, v in label_mapping.items()}


# === 工具函数 ===
def file_stem(name: str) -> str:
    """按最后一个点切分，移除后缀，返回文件名主体。
    例："a.b.c.txt" -> "a.b.c"，"image001.json" -> "image001"。
    """
    base = os.path.basename(name)
    return base.rsplit('.', 1)[0]


# === 函数定义 ===
def read_gt_labels(file_path):
    """读取YOLO格式的真实标签"""
    labels = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            cls_id = int(line.split()[0])
            labels.append(cls_id)
    return labels


def read_pred_labels(file_path):
    """读取预测文件，兼容多种格式：
    1) 纯 JSON 列表：[{"cls": "..."}, ...]
    2) 多个 JSON 对象直接拼接：{"cls":...}{"cls":...}
    3) Markdown 代码块包裹的 JSON，并且结构为 {"results": [{"detections": [{"cls": ...}, ...]}]}
    返回按 label_mapping 映射后的类别 ID 列表。
    """
    with open(file_path, 'r') as f:
        raw = f.read()

    content = raw.strip()
    if not content:
        return []

    def try_parse_json(text: str):
        try:
            return json.loads(text), None
        except Exception as e:
            return None, e

    data = None
    err = None

    # 优先处理 Markdown 代码块 ```json ... ```
    if '```' in content:
        # 提取第一个代码块内容（json 或普通代码块）
        matches = re.findall(r"```(?:json)?\s*([\s\S]*?)```", content)
        for block in matches:
            candidate = block.strip()
            data, err = try_parse_json(candidate)
            if data is not None:
                break
        # 若仍未解析成功，继续走后续兼容逻辑

    # 尝试直接解析为 JSON（适配已是完整 JSON 的情况）
    if data is None:
        data, err = try_parse_json(content)

    # 处理多 JSON 对象拼接的情况：}{ -> },{ 包裹为数组
    if data is None:
        glued = "[" + content.replace("}{", "},{") + "]"
        data, err = try_parse_json(glued)

    if data is None:
        print(f"[WARN] 无法解析 {file_path}: {err}")
        return []

    # 统一抽取包含 "cls" 的条目
    items = []
    if isinstance(data, list):
        items = data
    elif isinstance(data, dict):
        # 兼容 {"results": [{"detections": [...]}]} 结构
        if 'results' in data and isinstance(data['results'], list):
            for res in data['results']:
                dets = res.get('detections', [])
                if isinstance(dets, list):
                    items.extend(dets)
        # 若顶层直接有 detections
        elif 'detections' in data and isinstance(data['detections'], list):
            items.extend(data['detections'])
        else:
            # 回退：扫描字典所有值中的 list，收集含有 'cls' 的对象
            for v in data.values():
                if isinstance(v, list):
                    for obj in v:
                        if isinstance(obj, dict) and ('cls' in obj or 'class' in obj):
                            items.append(obj)

    cls_ids = []
    for item in items:
        if not isinstance(item, dict):
            continue
        cls_name = item.get("cls") or item.get("class") or ""
        cls_name = str(cls_name).strip()
        if not cls_name:
            continue
        if cls_name in label_mapping:
            cls_ids.append(label_mapping[cls_name])
        else:
            print(f"[WARN] 未知类别 {cls_name} in {file_path}")
    return cls_ids


# === 主计算逻辑 ===
gt_files = sorted(os.listdir(gt_txt_folder))
pred_files = sorted(os.listdir(pred_txt_folder))

# 将预测文件按“去后缀后的文件名”建立索引，以便与 GT 按同名匹配
pred_index = {}
for pf in pred_files:
    full = os.path.join(pred_txt_folder, pf)
    if os.path.isdir(full):
        continue
    pred_index[file_stem(pf)] = full

y_true, y_pred = [], []

for fname in gt_files:
    gt_path = os.path.join(gt_txt_folder, fname)
    if os.path.isdir(gt_path):
        continue
    stem = file_stem(fname)
    pred_path = pred_index.get(stem)

    if not pred_path or not os.path.exists(pred_path):
        print(f"[WARN] 缺少预测文件: 去后缀同名 '{stem}.*' 未在预测文件夹中找到")
        continue

    gt_labels = read_gt_labels(gt_path)
    pred_labels = read_pred_labels(pred_path)

    # 如果预测与真实数量不匹配，则按较短的长度比较
    n = min(len(gt_labels), len(pred_labels))
    if n == 0:
        continue

    y_true.extend(gt_labels[:n])
    y_pred.extend(pred_labels[:n])

# === 计算准确率与报告 ===
if len(y_true) == 0:
    print("❌ 没有有效的标签数据，请检查路径或文件内容。")
    exit()

acc = (np.array(y_true) == np.array(y_pred)).mean()
print(f"\n✅ 总体准确率 (Accuracy): {acc:.4f}\n")

print("📊 分类报告 (Classification Report):")
# 只针对实际出现过的类别生成报告，避免 target_names 与实际类别数量不一致
labels_present = sorted(set(y_true) | set(y_pred))
target_names_present = [id_to_label[i] for i in labels_present]
print(classification_report(
    y_true,
    y_pred,
    labels=labels_present,
    target_names=target_names_present,
    digits=4,
    zero_division=0
))

# === 混淆矩阵 ===
labels = sorted(label_mapping.values())
cm = confusion_matrix(y_true, y_pred, labels=labels)

# 保存混淆矩阵为 CSV
np.savetxt("confusion_matrix.csv", cm, fmt="%d", delimiter=",")

# 绘制并保存混淆矩阵图像
plt.figure(figsize=(10, 8))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=[id_to_label[i] for i in labels],
    yticklabels=[id_to_label[i] for i in labels]
)
plt.xlabel("Predicted Class")
plt.ylabel("True Class")
plt.title("Confusion Matrix for Worm Detection")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=300)
plt.show()
