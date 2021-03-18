# RSNA_Lung_Yolo_Pytorch
# gxzy_demo

## YOLO RSNA Object Detection
将yolo用于胸片影像目标检测，数据源于Kaggle:https://www.kaggle.com/c/rsna-pneumonia-detection-challenge/data ，原始数据经过了预处理，dcm格式转为了jpg，并且划分了训练集（21148张）和测试集（5536张），并对原始的csv标签进行了处理，生成了RSNAtrain和RSNAtest两个txt文件。

### quick start
#### 环境
CUDA10.1、python3.6、pytorch1.1.0、GPU:2080Ti*1  
需要的库
- opencv
- visdom
- tqdm
#### train
1.建议运行先启动visdom:
python3 -m visdom.server  
2.下载处理好的数据集和对于的txt文件，放在当前目录下：链接：https://pan.baidu.com/s/1TQpavAV7stX5VVywK-lq_g 提取码：qpxg 
3.python3 train.py  
在训练时会在当前路径创建一个log日志，最后会生成两个pth，一个是loss最小的best.pth，一个是最终的yolo.pth
#### eval
1.如果没有进行过前面的train，请下载已经训练好的模型：链接：https://pan.baidu.com/s/1TQpavAV7stX5VVywK-lq_g 提取码：qpxg 
2.python3 eval_voc.py

#### perdict
1.如果没有进行过前面的train，请下载已经训练好的模型：链接：https://pan.baidu.com/s/1TQpavAV7stX5VVywK-lq_g 提取码：qpxg 
2.python3 predict.py  
会在predict_img文件夹输出预测结果
