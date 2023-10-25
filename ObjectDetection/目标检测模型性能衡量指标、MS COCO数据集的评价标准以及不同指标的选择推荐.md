
<center><b><font size=16>目标检测模型性能衡量指标、MS COCO 数据集的评价标准以及不同指标的选择推荐</font></b></center>

# 0. 引言

## 0.1 COCO 数据集评价指标

<div align=center>
	<img src=https://img-blog.csdnimg.cn/804fea2cf5024ceeb83dae949d5fc88a.png
	width=100%>
</div>

目标检测模型通过 pycocotools 在验证集上会得到 COCO 的评价列表，具体参数的含义是什么呢？

## 0.2 目标检测领域常用的公开数据集

1. [PASCAL VOC](http://host.robots.ox.ac.uk/pascal/VOC/)
2. [Microsoft COCO（MS COCO）](https://cocodataset.org/)

在 MS COCO 数据集出来之前，目标检测基本上用的是 PASCAL VOC 数据集，现在 MS COCO 非常流行。这两个数据集均有自己的评判标准。

## 0.3 COCO（Common Objects in Context，上下文中的常见对象）数据集简介

### 0.3.1 介绍

COCO 数据集是一个可用于图像检测（Image Detection），语义分割（Semantic Segmentation）和图像标题生成（Image Captioning）的**大规模数据集**。它有超过 330K 张图像（其中 220K 张是有标注的图像），包含
+ 150 万个目标
+ 80 个目标类别（object categories：行人、汽车、大象等）
+ 91 种材料类别（stuff categoris：草、墙、天空等）
+ 每张图像包含五句图像的语句描述
+ 且有 250, 000 个带关键点标注的行人

> MS COCO官网：https://cocodataset.org/#home

### 0.3.2 MS COCO 可以应用的任务

1. **目标检测（object detection）**：使用 bounding box 或者 object segmentation (也称为instance segmentation)将不同的目标进行标定。

2. **Densepose（密集姿势估计）**：DensePose 任务涉及同时检测人、分割他们的身体并将属于人体的所有图像像素映射到身体的3D表面。用于不可控条件下的密集人体姿态	估计。
   <div align=center>
		<img src=https://img-blog.csdnimg.cn/502deb566cbb4386b42060356e217586.png
		width=100%>
	</div>

3. **Key-points detection（关键点检测）**：在任意姿态下对人物的关键点进行定位，该任务包含检测行人及定位到行人的关键点。
	<div align=center>
		<img src=https://img-blog.csdnimg.cn/65674dceddda4fb4950966088e1b695f.png
		width=100%>
	</div>

4. **Stuff Segmentation（材料细分）**：语义分割中针对 stuff class 类的分割（草，墙壁，天空等）
	<div align=center>
		<img src=https://img-blog.csdnimg.cn/72843bedec764e61be2f6f87feeae8c7.png
		width=100%>
	</div>

5. **Panoptic Segmentation（全景分割）**：其目的是生成丰富且完整的连贯场景分割，这是实现自主驾驶或增强现实等真实世界视觉系统的重要一步。
	<div align=center>
		<img src=https://img-blog.csdnimg.cn/914dde508f2b4d899186febc9cd61468.png
		width=100%>
	</div>

6. **image captioning（图像标题生成）**：根据图像生成一段文字。
	<div align=center>
		<img src=https://img-blog.csdnimg.cn/871402ff6e8346728d6ce41a46f98b89.png
		width=100%>
	</div>

### 0.3.3 COCO 的 80 个类别

<div align=center>
	<img src=https://img-blog.csdnimg.cn/844ef01914bb43c58d5f5f1db8150dc5.png
	width=100%>
	<img src=https://img-blog.csdnimg.cn/e37a2608953444118ba73cbc8263c20f.png
	width=100%>
</div>

<div align=center>

| 编号 | 英文名称 | 中文名称 | 编号 | 英文名称 | 中文名称 | 编号 | 英文名称 | 中文名称 |
| ---- | ------------------- | ---------------- | ---- | ------------------- | ---------------- | ---- | ------------------- | ---------------- |
| 1 | person | 人 | 28 | boat | 船 | 55 | cup | 杯子 |
| 2 | bicycle | 自行车 | 29 | traffic light | 交通灯 | 56 | fork | 叉子 |
| 3 | car | 汽车 | 30 | fire hydrant | 消防栓 | 57 | knife | 刀 |
| 4 | motorcycle | 摩托车 | 31 | stop sign | 停车标志 | 58 | spoon | 勺子 |
| 5 | airplane | 飞机 | 32 | parking meter | 停车计时器 | 59 | bowl | 碗 |
| 6 | bus | 公共汽车 | 33 | bench | 长凳 | 60 | banana | 香蕉 |
| 7 | train | 火车 | 34 | bird | 鸟 | 61 | apple | 苹果 |
| 8 | truck | 卡车 | 35 | cat | 猫 | 62 | sandwich | 三明治 |
| 9 | boat | 船 | 36 | dog | 狗 | 63 | orange | 橙子 |
| 10 | traffic light | 交通灯 | 37 | horse | 马 | 64 | broccoli | 西兰花 |
| 11 | fire hydrant | 消防栓 | 38 | sheep | 羊 | 65 | carrot | 胡萝卜 |
| 12 | stop sign | 停车标志 | 39 | cow | 牛 | 66 | hot dog | 热狗 |
| 13 | parking meter | 停车计时器 | 40 | elephant | 大象 | 67 | pizza | 披萨 |
| 14 | bench | 长凳 | 41 | bear | 熊 | 68 | donut | 甜甜圈 |
| 15 | bird | 鸟 | 42 | zebra | 斑马 | 69 | cake | 蛋糕 |
| 16 | cat | 猫 | 43 | giraffe | 长颈鹿 | 70 | chair | 椅子 |
| 17 | dog | 狗 | 44 | backpack | 背包 | 71 | couch | 沙发 |
| 18 | horse | 马 | 45 | umbrella | 雨伞 | 72 | potted plant | 盆栽 |
| 19 | sheep | 羊 | 46 | handbag | 手提包 | 73 | bed | 床 |
| 20 | cow | 牛 | 47 | tie | 领带 | 74 | dining table | 餐桌 |
| 21 | elephant | 大象 | 48 | suitcase | 行李箱 | 75 | toilet | 厕所 |
| 22 | bear | 熊 | 49 | frisbee | 飞盘 | 76 | tv monitor | 电视监视器 |
| 23 | zebra | 斑马 | 50 | skis | 滑雪板 | 77 | laptop | 笔记本电脑 |
| 24 | giraffe | 长颈鹿 | 51 | snowboard | 单板滑雪 | 78 | mouse | 鼠标 |
| 25 | backpack | 背包 | 52 | sports ball | 运动球 | 79 | remote | 遥控器 |
| 26 | umbrella | 雨伞 | 53 | kite | 风筝 | 80 | keyboard | 键盘 |
| 27 | tie | 领带 | 54 | baseball bat | 棒球棍 |

</div>

# 1. 目标检测中常见的指标

<div align=center>
	<img src=https://img-blog.csdnimg.cn/fa158de481544a2ea13ed02ff4036573.png
	width=50%>
</div>

对于这样一张图片，怎样才能算检测正确呢？其中，绿色为 GT，红色为预测框。

+ IoU 大于指定阈值？
+ 类别是否正确？
+ confidence 大于指定阈值？

以上三点都是我们需要考虑的。


## 1.1 TP、FP、FN

### 1.1.1 定义

+ TP（True Positive）：预测正确的预测框数量 [**IoU > 阈值**]（同一个 GT 只计算一次）
+ FP（False Positive）：检测到是同一个 GT 的多余预测框的数量 [**IoU < 阈值**]（或者是检测到同一个 GT 的多余预测框的数量）
+ FN（False Negative）：没有检测到 GT 的预测框数量 [**漏检的数量**]

> 阈值根据任务进行调整，一般选择 0.5
> FP 就是“假阳性”，就是模型**误认为**是 TP

### 1.1.2 例子说明TP、FP、FN

举个例子理解TP、FP、FN：
![在这里插入图片描述](https://img-blog.csdnimg.cn/a9c4a45400334ed382295c0aa32db51f.png)
对于张图片来说，<font color='green'>绿色</font>为GT，<font color='red'>红色</font>为模型预测框，IoU阈值设置为0.5。
+ 对于中间这只猫来说，在绿色框（GT）中的预测框（红色）和绿色框的IoU肯定是>0.5的，所以它应该是一个TP（预测对目标且IoU > 阈值）；而对于偏左的预测框来说，它和GT的IoU肯定是不足0.5的，加之因为有TP的存在，所以它是FP。
+ 对于右下角的那只猫，GT是有的，但模型并没有给出对应的预测框，因此模型对于这只猫来说，漏检了，故FN的数量为1。

## 1.2 AP（Average Precision，平均精度）
### 1.2.1 Precision

+ $\mathrm{Precision = \frac{TP}{TP + FP}}$：模型预测的所有目标中，预测正确的比例	-> ==查准率==
> 模型认为正确的目标中真的对了多少

---

那么仅仅通过Precision这个指标能不能全面衡量模型的检测能力呢？

举个例子进行说明：

![在这里插入图片描述](https://img-blog.csdnimg.cn/cda6d9a7f9ea4cc2bdfbb1a650d0a8ab.png)
上面这张图片有5个目标，但是网络只针对猫①给出了预测框，剩下的猫都没有检测出来。这里的TP=1，FP=0。所以此时的Precision为：
$$
\begin{aligned}
\mathrm{Precision} & = \mathrm{\frac{TP}{TP+FP}} \\
& = \frac{1}{1 + 0} \\
& = 1\\
& = 100\%
\end{aligned}
$$

很明显对于这张图片网络漏检了4个目标，所以仅仅通过Precision无法评判检测网络的性能。因此引入了另外一个指标——Recall。

### 1.2.2 Recall

+ $\mathrm{Recall = \frac{TP}{TP + FN}}$：所有真实目标中，模型预测正确的比例	-> ==查全率==
> 本应该检测对的，模型检测对了多少

那么我们只使用Recall这个指标来判定模型的好坏吗？

举个例子说明：

![在这里插入图片描述](https://img-blog.csdnimg.cn/e27bb97a30214974a185b1eee97a124e.png)
这张图片和上一张图片类似，而网络总共预测出了50个预测框（即50个目标）。这50个预测框中包括了所有要检测的目标，那么该网络针对这张图片的Recall为：

$$
\begin{aligned}
\mathrm{Recall} & = \mathrm{\frac{TP}{TP+FN}} \\
& = \frac{1}{1 + 0} \\
& = 1\\
& = 100\%
\end{aligned}
$$

很明显，单单使用Recall无法评判模型的好坏。

所以我们需要同时使用Precision和Recall这两个指标来进行网络性能的评判，引入——AP。


### 1.2.3 AP —— P-R曲线下面积
AP就是P-R曲线下方的面积，而P-R分别为Precision和Recall。

假设模型已经训练完毕，验证集为下面3张图片：
![在这里插入图片描述](https://img-blog.csdnimg.cn/bb67f09db0eb41a7806c0d97604b106f.png)
#### 1.2.3.1 第一张图片
![在这里插入图片描述](https://img-blog.csdnimg.cn/7af3458de62b48a2960ea7a8b6426558.png)
首先判断该图片中有几个目标（GT），很明显绿色的框有两个，所以有两个GT，即num_ob = 0 + 2 = 2。
接下来同一个列表统计网络所检测到的目标信息：
|GT id|Confidence|OB(IoU=0.5)|
|--|--|--|
|<font color='orange'>1</font>|<font color='orange'>0.98|<font color='orange'>True|
|<font color='orange'>1|<font color='orange'>0.61|<font color='orange'>False|

> 其中GT id为**预测框匹配的GT**的id，Confidence为预测框的置信度（为此类别的概率），OB为判断该预测框是否是TP。
> ==该表从上到下的顺序是根据Confidence降序排列的==
> 对于GT id=2，网络并没有给出预测框，所以表中没有

#### 1.2.3.2 第二张图片
![在这里插入图片描述](https://img-blog.csdnimg.cn/7444f1e3d139400383652e1c67073a84.png)
这张图片中目标的个数（绿色的框）有1个，所以累积目标个数num_ob = 2 + 1 = 3。

表也需更新：

|GT id|Confidence|OB(IoU=0.5)|
|--|--|--|
|1|0.98|True|
|<font color='orange'>3|<font color='orange'>0.89|<font color='orange'>True|
|<font color='orange'>3|<font color='orange'>0.66|<font color='orange'>False|
|1|0.61|False|

#### 1.2.3.3 第三张图片
![在这里插入图片描述](https://img-blog.csdnimg.cn/37760c546a2b45b0b67823709793f37d.png)
累积目标个数num_ob = 3 + 4 = 7。
更新表：

|GT id|Confidence|OB(IoU=0.5)|
|--|--|--|
|1|0.98|True|
|3|0.89|True|
|<font color='orange'>6<font color='orange'>|<font color='orange'>0.88|<font color='orange'>True|
|<font color='orange'>7|<font color='orange'>0.78|<font color='orange'>True|
|3|0.66|False|
|1|0.61|False|
|<font color='orange'>4|<font color='orange'>0.52|<font color='orange'>True|


#### 1.2.3.4 计算AP
得到表以后，我们计算针对不同Confidence（即取不同Confidence阈值）得到的Precision和Recall的信息）。
|GT id|Confidence|OB(IoU=0.5)|
|--|--|--|
|1|0.98|True|
|3|0.89|True|
|6|0.88|True|
|7|0.78|True|
|3|0.66|False|
|1|0.61|False|
|4|0.52|True|

1. 首先将Confidence的阈值设置为0.98（Confidence≥0.98的目标才算匹配正确），只有一个预测框符合条件（表中的第一行）。
	+ TP = 1; FP = 0; FN = 6
	> 在Confidence≥0.98的条件下，TP=1没什么问题；FP=0是因为阈值的存在；FN=6是因为累积目标个数num_ob=7，所以 $\mathrm{FN=num\_ob - TP} = 7 - 1 = 6$。因此我们可以得到$\mathrm{Precision = \frac{TP}{TP + FP} = \frac{1}{1+0}=1}$和$\mathrm{Recall = \frac{TP}{TP + FN} = \frac{1}{1+6}=0.14}$
	> Note：这个TP; FP; FN是看那个表，就不区分什么第几张图片了，看表就可以。

2. 将Confidence阈值设置为0.89
	+ 此条件下，TP=2;FP=0;FN=num_ob-TP=7-2=5，我们可以得到Precision和Recall

3. ...
4. 将Confidence阈值设置为0.66
	+ 此条件下，TP=4;FP=1;FN=num_ob-TP=7-4=3，我们可以得到$\mathrm{Precision = \frac{TP}{TP + FP} = \frac{4}{4+1}=0.80}$和$\mathrm{Recall = \frac{TP}{TP + FN} = \frac{4}{4+3}=0.57}$
5. ...

全部计算完毕后，结果如下表所示。

|Rank|Precision|Recall|
|--|--|--|
|1|1.0|0.14|
|2|1.0|0.28|
|3|1.0|0.42|
|4|1.0|0.57|
|5|0.80|0.57|
|6|0.66|0.57|
|7|0.71|0.71|

我们可以根据求得的一系列的Precision和Recall绘制P-R曲线。以Recall为横坐标，Precision为纵坐标得到P-R曲线，如下图所示。
![在这里插入图片描述](https://img-blog.csdnimg.cn/3dc7ae58f90b4556b4889ca931726ec0.png)
在绘制P-R曲线时需注意：
对于Recall（横坐标）需要滤除一些重复数据（图中用框住的即为参与计算的点，有两个点没有被框，它俩不参与AP的计算）。根据表中的数据可知，Recall=0.57有3个值，此时需保留Precision最大的值，即：

|Rank|Precision|Recall|
|--|--|--|
|1|1.0|0.14|
|2|1.0|0.28|
|3|1.0|0.42|
|4|1.0|0.57|
|5|~~0.80~~ |0.57|
|6|~~0.66~~|0.57|
|7|0.71|0.71|

图中阴影部分的面积就是AP，计算如下（重复的Recall已经滤除）：
|Rank|Precision|Recall|
|--|--|--|
|1|1.0|0.14|
|2|1.0|0.28|
|3|1.0|0.42|
|4|1.0|0.57|
|6|0.71|0.71|

$$
\begin{aligned}
\mathrm{Recall} & = \sum_{i=1}^{\mathrm{Rank}} (\mathrm{Recall}_i -\mathrm{Recall}_{i-1}) \times \max(\mathrm{Precision}_{ i, ..., \mathrm{Rank}}) \\
\mathrm{Recall} & = \sum_{i=本行}^{\mathrm{Rank}} (\mathrm{Recall}_{本行} -\mathrm{Recall}_{上一行}) \times 本行及以下最大的\mathrm{Precision}
\end{aligned}
$$

根据公式可以求得阴影的面积，即AP为：
$$
\begin{aligned}
\mathrm{Recall} & = (0.14 - 0) \times 1.0 + (0.28 - 0.14) \times 1.0 + (0.42 - 0.28) \times 1.0 + (0.57 - 0.42) \times 1.0 + (0.71 - 0.57) \times 0.71 \\
& = 0.6694
\end{aligned}
$$

了解完AP后我们就可以进一步得到一个新的指标——mAP。

## 1.3 mAP(mean Average Precision，即各类别AP的平均值)
mAP就是各类别AP的平均值，计算公式如下：

$$
\mathrm{
mAP = \frac{1}{nc}\sum^{nc}_{i=1}AP_i
}
$$

其中nc为类别数。
## 1.4 注意事项
以上的TP、FP、FN都是经过NMS处理后得到的预测框。

# 2. MS COCO评价指标中每条数据的含义
![在这里插入图片描述](https://img-blog.csdnimg.cn/44ed716d1e0f4d6a8a7cad7c317e47d3.png)
COCO官网说明：https://cocodataset.com/#detection-eval

![在这里插入图片描述](https://img-blog.csdnimg.cn/a7fe796cea8f4285abf865df9e9ee697.png)
> ==虽然写的是AP，但实际上表示的是mAP==

## 2.1 Average Precision (AP)
1. **AP**：COCO的主要评价指标，设置的IoU阈值为`IoU = range(0.5, 1.00, 0.05)`共10个IoU的mAP的**均值**
	$$
	\mathrm{AP = \frac{1}{10}(mAP^{IoU=0.5} + mAP^{IoU=0.55} + mAP^{IoU=0.60} + mAP^{IoU=0.65} + mAP^{IoU=0.70} + mAP^{IoU=0.75} + mAP^{IoU=0.80} + mAP^{IoU=0.85} + mAP^{IoU=0.9} + mAP^{IoU=0.95})}
	$$
2. **AP^IoU=0.50^**：将IoU阈值设置为0.5得到的mAP值（就是上面我们举的例子），这个取值也是PASCAL VOC的评价指标
3. **AP^IoU=0.75^**：更加严格的标准（因为IoU的阈值越大，说明网络预测框与GT重合度越来越高 -> 目标的定位越来越准，这对网络来说是很难的）

## 2.2 Across Scales
1. **AP^small^**：[mAP针对小目标] 需检测目标（GT）的像素面积小于32^2^，则将其归为小目标 —— 衡量网络对于小目标的平均查准率
2. **AP^medium^**：[mAP针对中目标] 需检测目标（GT）的像素面积在32^2^和96^2^，则将其归为中目标 —— 衡量网络对于中等目标的平均查准率
3. **AP^large^**：[mAP针对大目标] 需检测目标（GT）的像素面积大于96^2^，则将其归为大目标 —— 衡量网络对于大目标的平均查准率

通过这三个指标可以看出该目标检测网络对于不同尺度目标的检测效果。如果我们的任务需要检测的目标都是较小的，我们应该更加关注与AP^small^参数而不是AP^large^

## 2.3 Average Recall (AR)
对于目标检测网络，在代码部分会限制每张图片最终预测框的数量，这里的max就是这个数量。如max=100，即每张图片最终预测100个预测框。而这里的AR^max=100^就表示在每张图片预测框阈值为100的情况下，平均的查全率是多少。

1. **AR^max=1^**：在每张图片预测框阈值为1的情况下，平均的查全率是多少
2. **AR^max=10^**：在每张图片预测框阈值为10的情况下，平均的查全率是多少
3. **AR^max=100^**：在每张图片预测框阈值为100的情况下，平均的查全率是多少


![在这里插入图片描述](https://img-blog.csdnimg.cn/5d004d65b9c3497ab1fac2849177d875.png)

从上图可以看到，AR^max=100^=64%，AR^max=10^=63.3%，AR^max=1^=45.2%。这说明取max取100和取10相差不大，说明模型训练时使用的数据集**每张图片**中目标（GT）的数目并不是很多，基本上在10左右；而当预测框数量限制在1时，它的AR仅为45.2%，说明每张图片的目标个数一般是＞1的。

## 2.4 AR Across Scales
与AP、AP across scales类似，AR across scales表示对应**不同目标尺度的AR**。

1. **AR^small^**：[AR针对小目标] 需检测目标（GT）的像素面积小于32^2^，则将其归为小目标 —— 衡量网络对于小目标的平均查全率
2. **AR^medium^**：[AR针对中目标] 需检测目标（GT）的像素面积在32^2^和96^2^，则将其归为中目标 —— 衡量网络对于中等目标的平均查全率
3. **AR^large^**：[AR针对大目标] 需检测目标（GT）的像素面积大于96^2^，则将其归为大目标 —— 衡量网络对于大目标的平均查全率

# 3. 各种指标的选择 —— 基于不同的任务
不同的任务需要使用不同的指标。

![在这里插入图片描述](https://img-blog.csdnimg.cn/44ed716d1e0f4d6a8a7cad7c317e47d3.png)
## 3.1 mAP
+ 于PASCAL VOC的mAP来说，**AP^IoU=0.50^**是要看的，因为它是PASCAL VOC的主要评价指标。
+ 对于MS COCO数据集来说，**AP**（第一行，**10个mAP的平均**）是要看的，因为它是MS COCO的主要评价指标。
+ 如果我们对**目标框定位精度要求较高**的话，我们可以关注**AP^IoU=0.75^**
+ 如果我们对**小目标检测要求比较高**的话，我们可以关注**AP^small^**，通过这个值可以了解网络对于小目标检测的平均查准率（整体情况）
+ 如果我们对**中目标检测要求比较高**的话，我们可以关注**AP^medium^**
+ 如果我们对**大目标检测要求比较高**的话，我们可以关注**AP^large^**

## 3.2 AR
主要关注下面两个指标：
+ **AR^max=10^**
+ **AR^max=100^**

如果它俩**AR**（平均查全率）相差很小的话， 可以减少网络预测框的个数 -> 提高目标检测的效率

# 参考
1. https://www.bilibili.com/video/BV1ez4y1X7g2?spm_id_from=333.999.0.0
2. https://blog.csdn.net/qq_44554428/article/details/122597358#t2
