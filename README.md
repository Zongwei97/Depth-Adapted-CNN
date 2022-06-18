# Depth-Adapted-CNN

Demo of [Depth-Adapted CNN for RGB-D Cameras](https://openaccess.thecvf.com/content/ACCV2020/papers/Wu_Depth-Adapted_CNN_for_RGB-D_cameras_ACCV_2020_paper.pdf) (ACCV2020 Oral)

[aCNN.py](https://github.com/Zongwei97/Depth-Adapted-CNN/blob/main/aCNN.py) contains the basic functions for computing depth adapted sampling position.

[model.py](https://github.com/Zongwei97/Depth-Adapted-CNN/blob/main/model.py) shows a VGG-16 adapted to depth.

# Extensive comparison

We also propose an extention of our conferece paper which is available here: 

[Depth-Adapted CNNs for RGB-D Semantic Segmentation](https://arxiv.org/abs/2206.03939)

We extensively compare our approach with the concurrent [SConv TIP21](https://arxiv.org/pdf/2004.04534.pdf) and [ShapeConv ICCV21](https://arxiv.org/pdf/2108.10528.pdf). 

We show that our approach can achieve better performance when all methods are under the same baseline.

The source code will be released soon.

# Other application

We show that depth offset can be useful for other segmentation task such as [salient object dectection](https://arxiv.org/pdf/2110.04904.pdf).
