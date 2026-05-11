# Training Notes & Lessons Learned

Guidewire Pose Estimation 训练过程中踩过的坑和得到的经验，供后续调参参考。

---

## 核心问题诊断流程

### 1. 过拟合测试（Overfit Test）

**永远先跑过拟合测试**：用 10 张图、无 augmentation、高 lr，训练几百 epoch。

- 如果 loss 降不到接近 0 → 模型或 loss 有 bug
- 如果 loss 降到 0 但某个指标卡住 → 模型架构有瓶颈

```python
# 关键参数
batch_size = 10 (全部放一个 batch)
lr = 1e-3 ~ 1e-2
epochs = 500-2000
augmentation = False
dropout = 0.0
```

### 2. 确认问题后再调参

不要一开始就疯狂调超参数。诊断顺序：
1. 过拟合测试 → 验证模型能力
2. 全训练集（无 aug）→ 看 train 是否下降
3. 对比 train/val gap → 确定是过拟合还是欠拟合
4. 加 augmentation → 看泛化是否改善

---

## 踩过的坑

### Bug 1: ResNet + Global Average Pooling 丢失空间信息

**症状**：过拟合测试中 position error 卡在 ~23px，无论怎么调 lr 都降不下去。Direction 能正常学到 0。

**原因**：ResNet 的 `avgpool` 将 (B, 512, 12, 12) 的特征图压成 (B, 512)，完全丢弃了空间位置信息。FC head 只能从 channel 统计量"猜"位置，精度有上限。

**解决**：改用 spatial-aware position head：
- 去掉 ResNet 的 avgpool + fc
- 直接在空间特征图上做 conv → soft-argmax 提取坐标
- Direction head 仍用 GAP（方向不需要精确空间信息）

**验证**：简单 CNN（无 pretrained）能过拟合到 0.3px，证明问题在 GAP 不在任务本身。

### Bug 2: Hungarian Matching 震荡

**症状**：train_swap_rate 始终 ~50%，模型完全不收敛。

**原因**：两根导丝间距只有 ~30px（归一化空间 ~0.03），模型预测稍有偏差就会导致匹配翻转，梯度方向反复切换。

**解决**：
- 在 dataset `__init__` 中按 x 坐标对两根导丝排序（wire0 = x 更小的）
- 去掉 loss 中的 Hungarian matching，直接用固定顺序
- 评估时仍做 matching（对齐预测到 GT，这个方向是正确的）

**注意**：如果用了几何 augmentation（水平翻转），翻转后 x 顺序会反转，需要在 augmentation 之后重新排序！

### Bug 3: Augmentation 后未重新排序

**症状**：有 augmentation 时 train error 高达 250px，模型完全不学；无 aug 时 train 能降到 22px。

**原因**：水平翻转将 wire0（x 小的）变成 x 大的，但 wire index 没更新，模型收到了矛盾的标签。

**解决**：在 `_apply_augmentation()` 返回后，加一步：
```python
if positions[0, 0] > positions[1, 0]:
    positions = positions[[1, 0]]
    angles = angles[[1, 0]]
```

### Bug 4: Direction Loss 权重不足

**症状**：Position error 正常下降，但 angular error 始终 ~90°（等同随机）。

**原因**：position_loss_weight=5.0, direction_loss_weight=1.0。位置 loss（MSE on [0,1]）数值小（~0.001-0.01），方向 loss 主导了 total loss 但 direction head 从 GAP 特征中学不到有用信号。

**解决**：position_weight=5.0, direction_weight=5.0，给方向学习足够的梯度。

### Bug 5: Warmup + Early Stopping 冲突

**症状**：模型训练几个 epoch 就 early stop，best epoch 在 warmup 期间。

**原因**：Warmup 期间 val loss 自然下降（从随机初始化开始），warmup 结束后 lr 跳到 full 值引起震荡，early stopping 判定 "不再改进" 就停了。

**解决**：小数据集去掉 warmup（设 warmup_epochs=0），或让 early stopping 在 warmup 结束后才开始计数。

### Bug 6: Augmented data 上的 eval 指标是假的

**症状**：带 aug 训练时 train error 始终 ~250px，看似模型没学到任何东西。

**原因**：`positions_px` 存的是 augmentation 前的原始坐标，但图像已经被翻转/旋转了。用原始坐标评估变换后的预测，结果当然是垃圾值。

**解决**：评估时必须用 **无 augmentation 的 dataloader**（即使模型是用 aug 训练的）。

### Bug 7: TTA（Test-Time Augmentation）破坏 wire 排序

**症状**：关掉 TTA 前角度误差 ~82°，关掉后降到 ~20°。位置误差也受影响。

**原因**：TTA 做水平翻转后取平均，但翻转后两根导丝的 x 顺序反转了。TTA 的 un-flip 只翻转了坐标，没有交换 wire index，导致 wire0 的预测和 wire1 的预测被平均在一起。

**解决**：关掉 TTA（`use_tta=False`）。如果要用 TTA，需要在 un-flip 后重新按 x 排序 wire。

---

## 最终配置（已验证）

```python
# Data
input_size = 384
augment_train = False  # 几何 aug 关闭，用 mixup 代替

# Model
backbone = resnet18 (或 resnet34 效果更好)
pretrained = True
position_head = conv(C→256) + conv(256→K) + soft-argmax  # 空间感知，不用 GAP
direction_head = GAP + FC(C→256→K*2) + Tanh + L2_normalize
dropout = 0.3
freeze_backbone_epochs = 0  # 小数据集不冻结

# Training
optimizer = AdamW
lr = 1e-3, backbone_lr_scale = 0.1  # backbone lr = 1e-4
weight_decay = 1e-3
scheduler = CosineAnnealing(T_max=200)
warmup = 0
use_mixup = True, mixup_alpha = 0.4  # 关键正则化手段

# Loss
position_loss = MSE, weight = 5.0
direction_loss = cosine, weight = 5.0
Hungarian matching = 关闭（训练时），保留（评估时）

# Wire ordering
dataset 中按 x 坐标排序，augmentation 后重新排序

# Evaluation
TTA = 关闭
early_stopping_patience = 30
```

---

## 最终性能（test set, N=50）

| 实验 | Pos Mean (px) | Pos Median (px) | Ang Mean (°) | Ang Median (°) |
|------|:---:|:---:|:---:|:---:|
| **ResNet-34** | **105** | **87** | **18.5** | **10.0** |
| Heatmap Model | 109 | 98 | 20.1 | 16.2 |
| ResNet-50 | 122 | 113 | 25.3 | 19.3 |
| ResNet-18 (Baseline) | 124 | 91 | 25.4 | 15.2 |
| No Pre-training | 160 | 123 | 23.6 | 22.7 |

### 对比改进前

| 指标 | 改进前 | 改进后 (ResNet-34) | 提升 |
|------|:---:|:---:|:---:|
| Position Mean | 232px | 105px | **55%** |
| Position Median | 199px | 87px | **56%** |
| Angle Mean | 55° | 18.5° | **66%** |
| Angle Median | 37° | 10.0° | **73%** |

---

## 各改动的贡献分析

| 改动 | 影响 | 重要程度 |
|------|------|:---:|
| Spatial position head (去掉 GAP) | 位置精度从 ~230px 降到 ~120px | 最关键 |
| 固定 wire 排序 + 去掉 matching | 消除 50% swap rate，训练稳定 | 关键 |
| Mixup (alpha=0.4) | 减缓过拟合，val 从 ~120px 降到 ~105px | 重要 |
| Direction loss weight 5→5 | 角度从 ~90° 降到 ~20° | 关键 |
| 关闭 TTA | 角度从 ~82° 降到 ~20°（评估修复） | 关键 |
| 关闭几何 augmentation | 避免标签混乱，训练能正常收敛 | 重要 |
| 关闭 warmup | 防止 early stopping 过早触发 | 次要 |

---

## 泛化瓶颈分析

- 数据集只有 314 张图（train=190），这是最大的限制
- Block-split 确保 train/val 来自不同连续序列，减少数据泄漏
- 模型容量足够（能过拟合到 0.3px），泛化是主要瓶颈
- 特征图分辨率 12×12（input_size=384 时），每个 grid cell ~81px
- ResNet-34 > ResNet-18 > ResNet-50：中等容量配 mixup 最佳

## 未来改进方向

1. **更多数据 / 更好的 split**：如果能获取更多图像，或使用 leave-one-subject-out split
2. **FPN / 多尺度特征**：融合 layer2-4 特征图，提高空间分辨率
3. **修复 TTA**：在 un-flip 后重新按 x 排序 wire，就能安全启用 TTA
4. **几何 aug + 重排序**：虽然我们加了 post-aug 排序，但实测效果不如 mixup；值得在更大数据集上重试
5. **Curriculum learning**：先无 mixup 学到接近收敛，再开 mixup fine-tune
