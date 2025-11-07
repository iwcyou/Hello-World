from ultralytics import YOLO
import torch
import os

# ----------------------
# 环境检测
# ----------------------
num_gpus = torch.cuda.device_count()
print(f"Detected {num_gpus} GPUs: {[torch.cuda.get_device_name(i) for i in range(num_gpus)]}")

DATA_PATH = '/data/kunfeng/all_in_one_yolo_split/data.yaml'
SAVE_DIR = '/data/kunfeng/yolo_runs'

# ----------------------
# 1️⃣ 加载预训练 YOLOv8x 模型（最大模型）
# ----------------------
model = YOLO('yolov8x.pt')  # 自动下载预训练权重

# ----------------------
# 2️⃣ 开始训练
# ----------------------
results = model.train(
    data=DATA_PATH,
    epochs=100,
    batch=64,                   # 每GPU自动分配 batch/num_gpus
    imgsz=1024,                 # 显微镜高分辨率，增大输入尺寸
    workers=16,                 # 利用多线程数据加载
    device=[0, 1],              # 双GPU训练
    project=SAVE_DIR,
    name='parasite_detection_hpc',
    save_period=5,              # ✅ 每隔 5 个 epoch 额外保存一次 epochX.pt (epoch0,5,10,...)
    cache=True,                 # ✅ 使用缓存（内存够）
    pretrained=True,
    optimizer='AdamW',          # 更稳定的优化器
    lr0=0.0005,                 # 初始学习率（适合大batch）
    momentum=0.9,
    weight_decay=0.0005,
    amp=True,                   # 自动混合精度
    patience=20,                # 早停策略
    # imgsz_max=1280,             # 允许动态调整
    cos_lr=True,                # 余弦学习率调度
    verbose=True
)

# ----------------------
# 3️⃣ 在测试集上验证
# ----------------------
metrics = model.val(
    data=DATA_PATH,
    split='test',
    imgsz=1024,
    device=[0, 1],
    save_json=True,    # 导出COCO格式结果
    # save_hybrid=True   # 保存预测图片
)

# ----------------------
# 4️⃣ 输出指标
# ----------------------
print("\n✅ Evaluation results:")
print(f"mAP@0.5: {metrics.box.map50:.4f}")
print(f"mAP@0.5:0.95: {metrics.box.map:.4f}")
print(f"Precision: {metrics.box.p:.4f}")
print(f"Recall: {metrics.box.r:.4f}")

# ----------------------
# 5️⃣ 导出模型
# ----------------------
best_model_path = model.export(format='pt', dynamic=True)
print(f"✅ Best model saved to: {best_model_path}")

# ----------------------
# 6️⃣ 绘制混淆矩阵与 per-class mAP 曲线
# ----------------------
import importlib
import numpy as np

conf_matrix = metrics.confusion_matrix.matrix

def _plot_confusion_matrix(conf_matrix, names, save_dir):
    try:
        plotting = importlib.import_module('ultralytics.utils.plotting')
        func = getattr(plotting, 'plot_confusion_matrix', None)
        if func is None:
            print("[WARN] plot_confusion_matrix not available in this Ultralytics version; skipping plot.")
            return
        func(conf_matrix, names=names, save_dir=save_dir)
    except Exception as e:
        print(f"[WARN] Failed to plot confusion matrix: {e}")

_plot_confusion_matrix(conf_matrix, names=list(model.names.values()), save_dir=f"{SAVE_DIR}/parasite_detection_hpc")

# 每类 mAP 输出
print("\n📊 Per-class mAP@0.5:")
for cls_name, m in zip(model.names.values(), metrics.box.maps):
    print(f"{cls_name}: {m:.4f}")
