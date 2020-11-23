## 数据读取 data.py
* __load_data(data_path)__  
功能：读取数据集中的所有图片和标注，返回三个数组：  
*temp_img* 原版图片 每一项为400\*400\*3的array  
*trgt_img* 目标图片  每一项为400\*400\*3的array  
*label*  标注，每一项为一个字典，格式为：{"flaw_class":\<int\>, "bbox":[\<int\>,\<int\>,\<int\>,\<int\>]}  

## 数据预处理 prepross.py
* __cut(img, bbox)__  
功能：根据标注边框进行裁剪,返回裁剪后的局部图片  
* __resize(img, size)__  
功能：对图片进行resize，转换为size\*size的大小  
* __diff(img1, img2)__  
功能：返回两张图片差分后的图片
* __prepross(img1, img2, bbox, size)__  
功能：对一对样本图片进行预处理，返回裁剪并求差分后的图片  

## 多层前馈神经网络  
（待编写）  
## 卷积神经网络  
（待编写）  
