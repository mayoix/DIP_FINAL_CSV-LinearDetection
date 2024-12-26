# DIP_FINAL_CSV-LinearDetection
<br>

## Coding part 代码部分
### HSV_Detection_Original.m:
##### 基于HSV颜色空间，复现并优化了一种自适应分割直线检测算法。该算法通过提取目标区域的HSV像素值动态调整阈值，实现图像分割的自适应性；结合闭运算去噪及MAD最小二乘法进行直线拟合，提升检测精度。
#### Based on the HSV color space, an adaptive segmentation line detection algorithm is reproduced and optimized. The algorithm dynamically adjusts the threshold by extracting the HSV pixel value of the target area to achieve the adaptability of image segmentation; combined with closed operation denoising and MAD least squares method for line fitting, the detection accuracy is improved.
<br>

### HSV_Detection.m:
##### 在复现基础上提出改进，包括优化阈值更新策略以适应场景动态变化，设计目标丢失处理机制以增强鲁棒性，并改进形态学处理与拟合步骤以提升实时性与精度。
#### Improvements are proposed based on the reproduction, including optimizing the threshold update strategy to adapt to the dynamic changes of the scene, designing the target loss processing mechanism to enhance robustness, and improving the morphological processing and fitting steps to improve real-time performance and accuracy.
<br>

### 1-5.png:
##### 测试用例
#### Test samples
<br>

## result part 结果部分
### compare_recon_closed.fig:
##### 形态学图像预处理改进前后效果对比图
#### Comparison of morphological image preprocessing effects before and after improvement
<br>

### Huber & MAD.fig:
##### 直线拟合部分效果对比图
#### Comparison of effects of straight line fitting
<br>

## declaration
### 本项目为2024年秋季学期北京师范大学人工智能学院数字图像处理期末大作业，未经允许不得擅自使用。
<br>

### 项目成员：
#### ZRX：代码复现自适应分割及图像预处理部分；形态学算法改进；论文同上部分及排版；GitHub维护。
#### GXF：代码复现直线拟合部分；Huber回归方法改进；论文同上部分。
#### HXJ：动态调整阈值改进，论文同上部分。

<br>

## References
#### 尉金强,杜文正,孙晓艳,等.基于目标HSV（色调-饱和度-亮度）空间图像自适应分割的直线检测算法[J].火箭军工程大学学报,2024,38(05):59-68.
<br>
<br>

#### 2024.12.26
