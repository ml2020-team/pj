## 更新后的数据集

各类别样本数量
 1(逃花): 143
 2(塞网): 254
 3(破洞): 10
 4(缝头): 319
 5(水渍): 9
 6(脏污): 47
 7(白条): 8
14(未对齐): 562
16(伪色): 7
17(前后色差): 25
20(模板取错): 197
21(漏浆): 7
22(脱浆): 16
23(色纱): 7
24(飞絮): 23

瑕疵块最大边长2048，最小边长51

## __load_data__
 * __load_data(data_path=r'./data/cutted_data',size = 64, class_list = '[[1], [2], [14]]', process_mod='diff', resize_mod='padding', augmentate = True)__
 
 __返回内容：__
 
 return train_X, train_Y, test_X, test_Y
 
 __参数释义：__
 
 * *__data_path__* :数据路径
 
 * *__size__* :将裁剪后大小不一的图片统一缩放为size\*size\*3
 
 * *__class_list__* :分类方式,格式为字符串。
 
 　　　__例__：
 
 　　　* *__class_list = '[[1], [2], [14]]'__*
 
 　　　表示type1，type2, type14各为一类
 
 　　　* *__class_list = '[[1,2,3], [4,5], [14]]'__*
 
　 　　表示type1,type2,type3合并为一类，type4，type5合并为一类，type14单独为一类。
 
* *__process_mod__*:预处理模式：

　　　　__可选项：__

　　　　* *__'diff'__* :对两张图片进行绝对值差分（absdiff，不会出现负数）

　　　　* *__'trgt'__* :直接使用裁剪后的瑕疵图

　　　　* *__'max_pool'__* :先对两张图片进行最大池化，再差分

　　　　* *__'avg_pool'__* :先对两张图片进行平均池化，再差分
    
* *__resize_mod__*:resize模式：

　　　　__可选项：__

　　　　* *__'scale__* :将图片缩放为size\*size大小。

　　　　* *__'padding'__* :图片比例保持不变，周围填充黑色像素点。
    
* *__augmentate__*:是否使用数据增强，为Ture则使用数据增强后的新增数据，否则只使用原数据集中的图片。


## 数据预处理

#### data_augment.py

* read_img(img)

  功能：将图片转换成numpy数组

* cut(img, bbox)

  功能：根据标注边框进行裁剪,返回裁剪后的局部图片

* diff(img1, img2)

  功能：返回两张图片的numpy数组差分后的numpy数组

* process(img1, img2, bbox)

  功能：对一对样本图片进行裁剪

* mkdir(root_path, flaw_type)

  功能：删除已经存在的数据目录，新建空的数据目录，```flaw_type```是瑕疵类别集合

* catagory(flaw_type)

  功能：把原始数据集的每对样本图片（模板图和瑕疵图）用```process```处理，然后把差分图按照瑕疵类别分类存放。对于瑕疵类别 x，把该类的差分图存进```args.catagory_cut_raw_data_path/typex```目录下

* split_data(flaw_type, file_path)

  功能：在```catagory```原始数据集后，划分训练集和测试集，```file_path```是划分后的训练集(```train.txt```)和测试集(```test.txt```)的存放路径（这里是按照文件名划分数据集的）。存储的内容是```list```，```list```每个元素的格式为：```[flaw_type, path]```。```flaw_type```是瑕疵类别，```path```是文件名。根据瑕疵类别和文件名就能确定一个差分图文件。例如```[1, pic1.jpg]```表示差分图文件路径```args.catagory.../type1/pic1.jpg```

* pre_aug(flaw_type, file_path)

  功能：做增强数据集前的准备工作，创建目录，把训练集数据分类存储到```args.augmentated_data_path/```目录下

* aug_collection(flaw_type, flaw_count)

  功能：对每类瑕疵的训练集调用```augmentation.py```文件中的```transform_image```进行数据增强，把每类瑕疵的训练集样本数增加到```pic_number```。程序运行完，对于瑕疵类别x，```args.aug.../typex/```目录下有```pic_number```个图片

* conv2numpy(flaw_type, file_path)

  功能：从```args.augmentated_data_path/```读出增强后的训练集数据，根据```test.txt```从```args.catagory_cut_raw_data_path/```读出测试集数据，把训练集和测试集的数据图片```resize```后转换成numpy数组存储起来。最后，对于任务```x```(1,2,3)，```file_path/taskx/```路径下存放有训练集数据```x_train.npy```，```y_train.npy```，测试集数据```x_test.npy```，```y_test.npy```。此时瑕疵类别已经转换成0, 1, ..., 14，而不是1, 2, ..., 24了

#### augmentation.py

* random_flip_left_right(image)

  功能：随机左右翻转图片

* random_flip_up_down(image)

  功能：随机上下翻转图片

* random_contrast(image, minval=0.6, maxval=1.4)

  功能：随机改变图片的对比度

* random_brightness(image, minval=0., maxval=.2)

  功能：随机改变图片的亮度

* random_saturation(image, minval=0.4, maxval=2.)

  功能：随机改变图片的饱和度

* random_hue(image, minval=-0.04, maxval=0.08)

  功能：随机改变图片的色调

* tf_rotate(input_image, min_angle = -np.pi/2, max_angle = np.pi/2)

  功能：把图片随机旋转一个角度

* transform_image(image)

  功能：把一个图片使用上面的所有方法（外加一个随机旋转90度）进行变换

* 说明：对于样本数量较多的类别[1, 2, 4, 14, 20]，没有使用上述所有的数据增强方法

经过数据预处理，在```file_path/taskx/```路径下存放着任务```x```所需的训练集和测试集的```.npy```数据文件

## 算法介绍

### 简单cnn

#### model_simple_cnn.py



### 逻辑回归

#### model_logistic.py



## 实验结果

|               |                           简单cnn                            |   逻辑回归   | DecisionTree | Resnet-18 | mlp |
| :-----------: | :----------------------------------------------------------: | :----------: | :-----:| :-------: | :-----: |
| task1(3分类)  |       acc:0.79 class0:0.65 class1:0.84 class2:0.79       | acc:0.56 | 0.612| 0.70 | 0.63 |
| task2(5分类)  | acc:0.74 class0:0.65 class1:0.83 class2:0.86 class3:0.68 class4:0.68 | acc:0.48 |0.418| 0.65 |
| task3(15分类) |                           结果太差                           |   结果太差   |0.265|

