# CRTrack_my1 README

## 快速看懂数据

`CRTrack_my1` 是一个以 **scene + clip + 3 个 view** 组织的多视角目标追踪伪标签数据集。
如果你只想先快速理解数据，先看下面这几类核心文件：

```text
CRTrack_my1/
└── CRTrack_In-domain/
    ├── images/train/{scene}/{clip}/{scene}_View{1|2|3}/
    │   └── 图像序列
    │
    ├── images/train/{scene}/{clip}/bbox/{scene}_View{1|2|3}_gt.txt
    │   └── 单视角 bbox 伪标签
    │      格式: frame_id, local_id, x, y, w, h, score
    │
    ├── images/train/{scene}/{clip}/{scene}_View{1|2|3}_reprompt_rle.pkl
    │   └── 单视角 mask 伪标签
    │      格式: data[frame][obj_id] = {"rle": ..., "score": ...}
    │
    ├── ids_with_text_cross_view/{scene}/{clip}/{scene}_clipxx_id_match_texts.csv
    │   └── 原始跨视角全局ID-局部ID对应文件
    │      格式: id, view1, view2, view3, text
    │
    ├── ids_with_text_cross_view_selected/{scene}/{clip}/{scene}_clipxx_id_match_texts.csv
    │   └── 筛选后的三视角完全对齐标签
    │      3 个视角都成功对齐
    │
    └── ids_with_text_cross_view_selected2/{scene}/{clip}/{scene}_clipxx_id_match_texts.csv
        └── 更宽松的筛选标签
           包含：
           1) 三视角都对齐的标签
           2) 仅两个视角对齐、另一个视角缺失的标签
              缺失视角的局部ID记为 -1
```

最重要的几个概念：

* `local_id` / `obj_id`：单视角局部 ID，只在当前 `scene + clip + view` 内有效
* `id`：跨视角全局 ID，只在当前 `scene + clip` 内有效
* 不同 `clip` 之间 ID **不相关**
* 同一 `scene` 的不同 `clip` 之间追踪 **不连续**
* `View1 / View2 / View3` 是固定三视角
* 文本描述 `text` 与跨视角全局 ID 绑定

---

# 1. 数据集简介

`CRTrack_my1` 是一个基于 `CRTrack` 扩展构建的 **三视角（three-view）跨视角目标追踪伪标签数据集**。
该数据集以 `scene + clip + view` 为基本组织单元，包含：

* 多视角图像序列
* 单视角 bbox 伪标签
* 单视角 mask 伪标签
* 跨视角全局 ID 与各视角局部 ID 的对应关系
* 每个跨视角目标对应的文本描述
* 不同筛选强度下的跨视角对齐标签

本数据集主要适用于以下研究任务：

* 多视角目标追踪（multi-view tracking）
* 跨视角目标关联（cross-view association）
* 文本引导目标追踪（language-guided tracking）
* mask tracking / segmentation
* 多视角伪标签构建与质量分析
* 结合视觉与文本的弱监督目标追踪研究

---

# 2. 数据组织原则

## 2.1 基本单位

数据按如下层级组织：

* **scene**：场景，例如 `Floor`、`Ground`
* **clip**：一个场景中的局部时间片段，例如 `clip_01`、`clip_02`
* **view**：固定 3 个视角

  * `View1`
  * `View2`
  * `View3`

## 2.2 多视角设置

每个 `scene` 下的每个 `clip` 都对应 3 个同步视角：

* `View1`
* `View2`
* `View3`

## 2.3 ID 独立性说明

本数据集中的 ID 具有如下约定：

### 局部 ID（local ID / obj_id）

* 出现在单视角 bbox 文件中，记为 `local_id`
* 出现在 mask pkl 文件中，记为 `obj_id`
* 仅在当前 `scene + clip + view` 内有效
* 不同视角之间不共享
* 不同 clip 之间不相关

### 全局 ID（global id）

* 出现在跨视角 csv 文件中，列名为 `id`
* 表示在同一个 `scene + clip` 下，同一人物在 `View1 / View2 / View3` 中的统一身份
* 仅在当前 `scene + clip` 内有效
* 不同 clip 之间不相关
* 同一 scene 不同 clip 之间追踪不连续

---

# 3. 根目录结构

数据根目录为：

```text
CRTrack_my1/
```

典型目录结构如下：

```text
CRTrack_my1/
└── CRTrack_In-domain/
    ├── images/
    │   └── train/
    │       ├── Floor/
    │       │   ├── clip_01/
    │       │   │   ├── Floor_View1/
    │       │   │   ├── Floor_View2/
    │       │   │   ├── Floor_View3/
    │       │   │   ├── Floor_View1_reprompt_rle.pkl
    │       │   │   ├── Floor_View2_reprompt_rle.pkl
    │       │   │   ├── Floor_View3_reprompt_rle.pkl
    │       │   │   └── bbox/
    │       │   ├── clip_02/
    │       │   └── ...
    │       ├── Ground/
    │       │   ├── clip_01/
    │       │   ├── clip_02/
    │       │   └── ...
    │       └── ...
    │
    ├── ids_with_text_cross_view/
    │   ├── Floor/
    │   ├── Ground/
    │   └── ...
    │
    ├── ids_with_text_cross_view_selected/
    │   ├── Floor/
    │   ├── Ground/
    │   └── ...
    │
    └── ids_with_text_cross_view_selected2/
        ├── Floor/
        ├── Ground/
        └── ...
```

---

# 4. 图像文件组织方式

## 4.1 图像路径示例

例如：

```text
/root/autodl-tmp/CRMOT/datasets/CRTrack_my1/CRTrack_In-domain/images/train/Floor/clip_02/Floor_View3/Floor_View3_000211.jpg
/root/autodl-tmp/CRMOT/datasets/CRTrack_my1/CRTrack_In-domain/images/train/Ground/clip_08/Ground_View1/Ground_View1_000463.jpg
```

## 4.2 命名规则

图像文件命名格式为：

```text
{scene}_View{n}_{frame_id:06d}.jpg
```

例如：

```text
Floor_View3_000211.jpg
Ground_View1_000463.jpg
```

其中：

* `{scene}`：场景名，如 `Floor`、`Ground`
* `View{n}`：视角名，`View1 / View2 / View3`
* `{frame_id}`：帧号，通常以 6 位数字补零表示

## 4.3 帧号说明

虽然图像按 `clip` 组织，但文件名中的帧号可以不是从 `1` 开始。
例如：

```text
Floor_View3_000211.jpg
```

说明该 clip 对应原始长序列中的一个局部时间片段，使用的是原始视频中的真实帧号。

因此：

* `clip` 是局部时序片段
* 图像名中的帧号是原始帧号
* 不同 clip 的帧号范围可能不连续

---

# 5. 单视角 bbox 伪标签

## 5.1 文件位置

bbox 标注文件位于每个 `scene/clip` 下的 `bbox/` 目录，例如：

```text
/root/autodl-tmp/CRMOT/datasets/CRTrack_my1/CRTrack_In-domain/images/train/Floor/clip_02/bbox/Floor_View1_gt.txt
/root/autodl-tmp/CRMOT/datasets/CRTrack_my1/CRTrack_In-domain/images/train/Floor/clip_02/bbox/Floor_View2_gt.txt
```

> 注意：文件名虽然为 `_gt.txt`，但这里保存的是**伪标签 bbox 结果**，并不一定是人工真实标注 GT。
> 该命名方式主要用于兼容部分现有训练或处理代码。

## 5.2 文件覆盖范围

每个 `{scene}_ViewX_gt.txt` 文件保存的是：

* 当前 `scene + view` 下
* 所有 clip 的 bbox 标注

也就是说，虽然文件位于某个 `clip` 目录中，但内容可能覆盖该 `scene + view` 的多个 clip 帧范围。
实际使用时，建议结合当前 clip 的帧范围进行筛选。

## 5.3 标注格式

每一行格式为：

```text
frame_id, local_id, x, y, w, h, score
```

字段含义：

* `frame_id`：帧号
* `local_id`：当前 `scene + clip + view` 下的局部 ID
* `x`：bbox 左上角横坐标
* `y`：bbox 左上角纵坐标
* `w`：bbox 宽度
* `h`：bbox 高度
* `score`：bbox 分数

## 5.4 格式示例

```text
1,0,0,415,15,77,1.000000
1,1,154,441,58,150,1.000000
1,2,468,369,48,104,1.000000
1,3,561,369,21,79,1.000000
1,19,886,463,63,133,2.203125
2,0,0,415,15,77,6.750000
2,1,155,441,57,149,8.562500
2,2,468,369,47,103,8.375000
2,3,562,373,19,72,5.187500
2,4,736,458,57,134,8.750000
```

## 5.5 bbox 坐标定义

bbox 采用：

```text
x, y, w, h
```

其中：

* `x, y` 为左上角坐标
* `w, h` 为宽和高
* 坐标单位为像素
* 不做归一化

---

# 6. Mask 伪标签

除 bbox 伪标签外，`CRTrack_my1` 还提供了基于 mask 的伪标签文件，用于实例分割、mask tracking、多视角区域建模等任务。

## 6.1 文件位置

mask 伪标签文件通常位于每个 `scene/clip` 目录下，例如：

```text
CRTrack_my1/CRTrack_In-domain/images/train/Floor/clip_03/Floor_View1_reprompt_rle.pkl
```

同一个 `clip` 下通常对应三个视角文件：

```text
CRTrack_my1/CRTrack_In-domain/images/train/{scene}/{clip}/{scene}_View1_reprompt_rle.pkl
CRTrack_my1/CRTrack_In-domain/images/train/{scene}/{clip}/{scene}_View2_reprompt_rle.pkl
CRTrack_my1/CRTrack_In-domain/images/train/{scene}/{clip}/{scene}_View3_reprompt_rle.pkl
```

## 6.2 数据格式

当前 mask 伪标签文件采用 pickle (`.pkl`) 格式存储，核心结构为：

```python
data[frame][obj_id] = {
    "rle": {"size": [H, W], "counts": bytes},
    "score": float
}
```

其中：

* `frame`：帧号
* `obj_id`：当前 `scene + clip + view` 下的局部 pseudo id
* `rle`：当前帧目标 mask 的 RLE 编码
* `score`：mask 置信度或质量分数

## 6.3 字段说明

### frame

`frame` 表示帧号，应与图像文件名中的帧号对应。
例如：

```text
Floor_View1_000211.jpg
```

其对应帧号为：

```text
211
```

在实际使用中，建议先打印键类型，确认 `frame` 是整数还是字符串。

### obj_id

`obj_id` 表示当前 `scene + clip + view` 下的局部 pseudo id。
通常应与对应视角 bbox 文件中的 `local_id` 对应。

它不是跨视角 global id，只在当前视角局部有效。

### rle

`rle` 使用 COCO 风格压缩表示：

```python
{
    "size": [H, W],
    "counts": bytes
}
```

其中：

* `size`：mask 尺寸，高和宽
* `counts`：压缩后的 RLE 编码，通常为 `bytes`

### score

`score` 表示该 mask 的置信度或质量分数。

## 6.4 作用范围

每个 `*_reprompt_rle.pkl` 文件对应：

* 一个 `scene`
* 一个 `clip`
* 一个 `view`

因此其作用域为：

```text
scene + clip + view
```

其中：

* 不同 view 间 `obj_id` 不共享
* 不同 clip 间 `obj_id` 不相关
* 若需要跨视角统一身份，需通过跨视角 csv 映射到 `global id`

## 6.5 读取 pkl 示例代码

```python
import pickle

pkl_path = r"CRTrack_my1/CRTrack_In-domain/images/train/Floor/clip_03/Floor_View1_reprompt_rle.pkl"

with open(pkl_path, "rb") as f:
    data = pickle.load(f)

print("top-level type:", type(data))
print("num frames:", len(data))

frame_keys = list(data.keys())[:5]
print("first few frame keys:", frame_keys)

frame = frame_keys[0]
print("frame =", frame)
print("objects in this frame:", list(data[frame].keys())[:10])

obj_id = list(data[frame].keys())[0]
obj_info = data[frame][obj_id]

print("obj_id:", obj_id)
print("keys:", obj_info.keys())
print("rle size:", obj_info["rle"]["size"])
print("score:", obj_info["score"])
print("counts type:", type(obj_info["rle"]["counts"]))
```

## 6.6 解码 RLE mask 示例

如果安装了 `pycocotools`，可以将 RLE 解码为二值 mask：

```python
import pickle
import numpy as np
from pycocotools import mask as mask_utils

pkl_path = r"CRTrack_my1/CRTrack_In-domain/images/train/Floor/clip_03/Floor_View1_reprompt_rle.pkl"

with open(pkl_path, "rb") as f:
    data = pickle.load(f)

frame = list(data.keys())[0]
obj_id = list(data[frame].keys())[0]

rle = data[frame][obj_id]["rle"]

if isinstance(rle["counts"], str):
    rle = {
        "size": rle["size"],
        "counts": rle["counts"].encode("ascii")
    }

mask = mask_utils.decode(rle)
mask = np.asarray(mask, dtype=np.uint8)

print("mask shape:", mask.shape)
print("mask unique values:", np.unique(mask))
```

## 6.7 与 bbox 标注的关系

mask 伪标签与 bbox 伪标签通常描述的是同一批目标，但存储形式不同：

* bbox 文件保存矩形框：

  * `frame_id, local_id, x, y, w, h, score`
* mask pkl 文件保存像素级区域：

  * `data[frame][obj_id]["rle"]`

通常情况下：

* `obj_id` 应与同视角 bbox 文件中的 `local_id` 对应
* 同一个目标在同一帧可同时具有 bbox 和 mask 信息
* 若需做 mask 可视化或 mask-to-box 转换，可先解码 RLE，再计算外接框

## 6.8 注意事项

1. `obj_id` 是局部 ID，不是跨视角 global id
2. `frame` 的键类型建议先检查，可能是 `int` 或 `str`
3. `counts` 可能是 `bytes` 或 `str`，解码前建议统一
4. 并不是所有目标都会在所有帧中出现，访问前建议判断键是否存在

---

# 7. 跨视角全局 ID 对应文件

## 7.1 原始跨视角对应文件位置

文件位于：

```text
CRTrack_my1/CRTrack_In-domain/ids_with_text_cross_view/{scene}/{clip}/{scene}_clip{xx}_id_match_texts.csv
```

例如：

```text
/root/autodl-tmp/CRMOT/datasets/CRTrack_my1/CRTrack_In-domain/ids_with_text_cross_view/Floor/clip_01/Floor_clip01_id_match_texts.csv
/root/autodl-tmp/CRMOT/datasets/CRTrack_my1/CRTrack_In-domain/ids_with_text_cross_view/Ground/clip_04/Ground_clip04_id_match_texts.csv
```

## 7.2 文件作用

该文件定义了：

* 当前 `scene + clip` 下的跨视角 global id
* 与 3 个视角局部 ID 的对应关系
* 该目标对应的文本描述

即：在同一个 clip 中，哪些 `View1 / View2 / View3` 的局部轨迹属于同一个人。

## 7.3 文件格式

CSV 字段为：

```csv
id,view1,view2,view3,text
```

字段含义：

* `id`：伪标签全局 ID
* `view1`：该全局 ID 在 `View1` 中对应的局部 ID
* `view2`：该全局 ID 在 `View2` 中对应的局部 ID
* `view3`：该全局 ID 在 `View3` 中对应的局部 ID
* `text`：该跨视角目标对应的文本描述

## 7.4 示例

```csv
id,view1,view2,view3,text
0,0,3,5,"A person in a black coat, dark trousers, brown shoes, holding a shopping bag."
1,1,8,4,"A person in a black coat, black trousers, white sneakers, holding a phone."
2,2,7,0,"A person with a black beanie, in a black coat, black trousers, dark sneakers."
3,3,4,0,"A person in a black coat, black trousers, black shoes."
```

## 7.5 重要说明

1. `id` 是当前 `scene + clip` 内的全局 ID
2. `view1/view2/view3` 是各视角局部 ID
3. 不同 clip 之间 ID 不相关
4. 同一 scene 不同 clip 间追踪不连续

---

# 8. 筛选后的跨视角标签

除了原始的 `ids_with_text_cross_view` 外，数据集还提供两类筛选后的标签。

## 8.1 `ids_with_text_cross_view_selected`

路径：

```text
/data/zhangxintian/school/data/CRTrack_my1/CRTrack_In-domain/ids_with_text_cross_view_selected
```

该目录中存放的是：

* **筛选好的三视角都对齐的标签**
* 即每一条记录在 `View1 / View2 / View3` 中都能找到对应局部 ID
* 更适合用于严格的三视角一致性实验

可以理解为：

> 这是“高质量、严格三视角完全对齐”的版本。

## 8.2 `ids_with_text_cross_view_selected2`

路径：

```text
/data/zhangxintian/school/data/CRTrack_my1/CRTrack-In-domain/ids_with_text_cross_view_selected2
```

该目录中存放的是：

* 筛选好的三视角都对齐的标签
* 以及部分只有两个视角成功对齐、另一个视角没有匹配上的标签

对于缺失匹配的视角：

* 该视角局部 ID 用 `-1` 表示

例如：

```csv
id,view1,view2,view3,text
10,3,7,-1,"A person in a dark coat and black trousers."
11,-1,2,5,"A person wearing a white coat and dark shoes."
```

可以理解为：

> 这是“较宽松”的版本，兼容三视角完全对齐样本，以及两视角对齐、单视角缺失样本。

## 8.3 两个目录的区别

### `selected`

* 只保留三视角全部对齐的记录
* 每条记录的 `view1/view2/view3` 都应为有效局部 ID
* 适用于高精度、严格三视角对齐实验

### `selected2`

* 包含 `selected` 中的三视角对齐记录
* 也包含“两视角对齐 + 单视角缺失”的记录
* 缺失视角用 `-1` 表示
* 适用于更宽松的数据利用方式

## 8.4 使用建议

* 若任务要求三视角严格一致，优先使用 `ids_with_text_cross_view_selected`
* 若希望利用更多样本、允许部分视角缺失，使用 `ids_with_text_cross_view_selected2`

---

# 9. 文本描述

## 9.1 文本描述来源

本数据集为每个跨视角 global id 提供一条文本描述，存储在跨视角 csv 文件的 `text` 列中。

文本主要描述目标人物的外观属性，例如：

* 上衣颜色
* 裤子颜色
* 鞋子
* 帽子
* 手持物
* 携带物
* 外观整体特征

## 9.2 文本格式示例

例如：

```text
A person in a black coat, dark trousers, brown shoes, holding a shopping bag.
A person in a black coat, black trousers, white sneakers, holding a phone.
A person with a black beanie, in a black coat, black trousers, dark sneakers.
```

## 9.3 与原始 CRTrack 文本的关系

原始数据集中的文本文件例如：

```text
/root/autodl-tmp/CRMOT/datasets/CRTrack/CRTrack_In-domain/labels_with_ids_text/train/gt_train/Floor.txt
```

格式通常为：

```text
<ID>:<text>
```

例如：

```text
1:A man wearing a black coat and black trousers.
2:A man wearing a white coat, black trousers and white shoes.
4:A man wearing a black coat, gray trousers and black shoes.
```

而在 `CRTrack_my1` 中，文本描述与每个 `scene + clip` 内的跨视角 global id 绑定，统一保存在：

```csv
id,view1,view2,view3,text
```

---

# 10. ID 定义总结

为了避免混淆，这里统一说明三类 ID。

## 10.1 单视角局部 ID

出现在 bbox 文件中：

```text
frame_id, local_id, x, y, w, h, score
```

以及 mask 文件中：

```python
data[frame][obj_id]
```

其中：

* `local_id` / `obj_id` 是单视角局部 ID
* 只在当前 `scene + clip + view` 内有效
* 不跨视角共享
* 不跨 clip 共享

## 10.2 跨视角全局 ID

出现在跨视角 csv 中：

```csv
id,view1,view2,view3,text
```

其中 `id`：

* 是当前 `scene + clip` 内的跨视角统一 ID
* 用于关联三视角中属于同一人的局部 ID
* 不跨 clip 共享
* 同一 scene 不同 clip 间也不连续

## 10.3 原始 GT / 参考数据中的 ID

如果在评估过程中使用原始 `CRTrack` 的 GT ID，需要注意：

* 原始 GT ID 不一定与 `CRTrack_my1` 的局部 ID 或全局 ID 一致
* `CRTrack_my1` 的 ID 体系是围绕伪标签构建流程定义的
* 二者关系应以额外匹配文件或评估脚本为准

---

# 11. 命名映射

若需要与其他代码库或旧版命名兼容，可使用如下映射关系：

* `scene1 -> Shop`
* `cam01 -> View1`
* `cam02 -> View2`

当前 `CRTrack_my1` 中推荐统一使用：

* 场景名：如 `Floor`、`Ground`
* 视角名：`View1`、`View2`、`View3`

---

# 12. 使用建议

本数据集可支持以下几类任务：

## 12.1 单视角目标追踪

使用 `bbox/{scene}_ViewX_gt.txt` 中的局部 ID 标注进行单视角训练或分析。

## 12.2 mask tracking / segmentation

使用 `{scene}_ViewX_reprompt_rle.pkl` 中的 mask 伪标签进行：

* 单目标 / 多目标 mask tracking
* 实例分割训练
* bbox-mask 一致性分析
* 可视化检查

## 12.3 跨视角身份关联

使用：

```csv
id,view1,view2,view3,text
```

进行：

* cross-view association
* multi-camera identity alignment
* multi-view consistency analysis

## 12.4 文本引导追踪

将 `text` 作为语言描述输入，研究：

* 文本驱动目标定位
* 语言引导跨视角检索
* 视觉-语言联合建模

## 12.5 伪标签质量分析

可基于：

* 单视角 bbox 伪标签
* 单视角 mask 伪标签
* 跨视角 ID 对应关系
* 文本描述一致性
* selected / selected2 两类筛选标签

开展伪标签质量评估、错误分析与可视化检查。

---

# 13. 读取示例

## 13.1 读取跨视角 CSV

```python
import pandas as pd

csv_path = r"/root/autodl-tmp/CRMOT/datasets/CRTrack_my1/CRTrack_In-domain/ids_with_text_cross_view/Floor/clip_01/Floor_clip01_id_match_texts.csv"
df = pd.read_csv(csv_path)

print(df.head())
```

## 13.2 读取 bbox 标注

```python
bbox_path = r"/root/autodl-tmp/CRMOT/datasets/CRTrack_my1/CRTrack_In-domain/images/train/Floor/clip_02/bbox/Floor_View1_gt.txt"

records = []
with open(bbox_path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        frame_id, local_id, x, y, w, h, score = line.split(",")
        records.append({
            "frame_id": int(frame_id),
            "local_id": int(local_id),
            "x": float(x),
            "y": float(y),
            "w": float(w),
            "h": float(h),
            "score": float(score),
        })

print(records[:5])
```

## 13.3 根据 global id 查找各视角 local id

```python
import pandas as pd

csv_path = r"/root/autodl-tmp/CRMOT/datasets/CRTrack_my1/CRTrack_In-domain/ids_with_text_cross_view/Floor/clip_01/Floor_clip01_id_match_texts.csv"
df = pd.read_csv(csv_path)

target_gid = 2
row = df[df["id"] == target_gid].iloc[0]

print("global id:", row["id"])
print("view1 local id:", row["view1"])
print("view2 local id:", row["view2"])
print("view3 local id:", row["view3"])
print("text:", row["text"])
```

## 13.4 读取 mask pkl

```python
import pickle

pkl_path = r"CRTrack_my1/CRTrack_In-domain/images/train/Floor/clip_03/Floor_View1_reprompt_rle.pkl"

with open(pkl_path, "rb") as f:
    data = pickle.load(f)

print("top-level type:", type(data))
print("num frames:", len(data))

frame_keys = list(data.keys())[:5]
print("first few frame keys:", frame_keys)

frame = frame_keys[0]
print("frame =", frame)
print("objects in this frame:", list(data[frame].keys())[:10])

obj_id = list(data[frame].keys())[0]
obj_info = data[frame][obj_id]

print("obj_id:", obj_id)
print("rle size:", obj_info["rle"]["size"])
print("score:", obj_info["score"])
```

## 13.5 解码 RLE mask

```python
import pickle
import numpy as np
from pycocotools import mask as mask_utils

pkl_path = r"CRTrack_my1/CRTrack_In-domain/images/train/Floor/clip_03/Floor_View1_reprompt_rle.pkl"

with open(pkl_path, "rb") as f:
    data = pickle.load(f)

frame = list(data.keys())[0]
obj_id = list(data[frame].keys())[0]
rle = data[frame][obj_id]["rle"]

if isinstance(rle["counts"], str):
    rle = {
        "size": rle["size"],
        "counts": rle["counts"].encode("ascii")
    }

mask = mask_utils.decode(rle)
mask = np.asarray(mask, dtype=np.uint8)

print("mask shape:", mask.shape)
print("mask unique values:", np.unique(mask))
```

---

# 14. 注意事项

## 14.1 `_gt.txt` 不一定是真实 GT

虽然文件名中包含 `_gt`，但在 `CRTrack_my1` 中这些文件主要存储的是 bbox 伪标签结果。
请不要直接将其视为人工精标 GT。

## 14.2 不同 clip 间身份不连续

即使属于同一 scene：

* `clip_01` 和 `clip_02` 中的人物 ID 也不保证连续
* 跨 clip 的 ReID 或长期追踪需要额外处理

## 14.3 局部 ID 与全局 ID 不能混用

* bbox / mask 文件中的 ID 是单视角局部 ID
* csv 文件中的 `id` 是 clip 内跨视角全局 ID

两者语义不同，应显式区分。

## 14.4 bbox 文件位置与逻辑范围可能不同

bbox 文件虽然位于某个 clip 目录中，但其内容可能覆盖当前 `scene + view` 下多个 clip 的帧。
因此在使用时，建议结合图像帧范围筛选当前 clip 对应记录。

## 14.5 `selected2` 中的 `-1` 表示缺失视角

在 `ids_with_text_cross_view_selected2` 中：

* 若某个视角没有成功匹配到局部 ID
* 则该视角位置记为 `-1`

因此处理该文件时，应显式判断是否存在缺失视角。

## 14.6 mask 中的 frame / counts 类型建议先检查

在使用 mask pkl 前，建议先检查：

* `frame` 是 `int` 还是 `str`
* `rle["counts"]` 是 `bytes` 还是 `str`

以避免后续解码出错。

---
