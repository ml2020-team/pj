## 数据集文件形式 
我们将预处理后的数据保存为.npy格式 

每个分类任务分别提供了3种图片大小和6种预处理方式

数据集文件结构如下：

task1

|-size=64

|-|-1_task1_trgt_padding

|-|-2_diff_scale

|-|-3_diff_padding

|-|-4_diff_padding_augment

|-|-5_max_pool_padding_augment

|-|-6_avg_pool_padding_augment

|-size=127

...

|-size=256

...

task2

...


## 预处理函数接口

### __读取图片并进行池化、差分、resize：load_data__
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


### 数据增强：

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

# 模型训练测试函数接口

### 简单cnn

#### model_simple_cnn.py



### 逻辑回归

#### model_logistic.py

### SVM

#### model_svm.py

### ResNet-18

#### model_resnet_18.py

### 决策树

#### model_decisionTree.py

