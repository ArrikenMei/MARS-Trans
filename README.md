# MARS-Trans: Multistage Attention Region Supplement Transformer for Fine-grained Visual Categorization

An improved model with Vision Transformer as the backbone called MARS-Trans is now used in Fine-grained Visual Categorization.

## Introduction
"MARS-Trans" is a variant model of the Vision Transformer proposed by the Big Data Analysis Lab of the School of Information Engineering, Henan University of Science and Technology.
<div align=left>
<img src='./docs/Figure2.png' width=900>
</div>
## DataSets
<table border="1" width="90%">
    <tr align="center">
        <th>Name</th><th>Class</th><th>Train</th><th>Test</th><th>Download</th><th>Size</th>
    </tr>
    <tr align="center">
        <th>CUB200-2011</th><th>200</th><th>5994</th><th>5794</th><th><a href="https://data.caltech.edu/records/65de6-vp158">https://data.caltech.edu/records/65de6-vp158</a></th><th>1.2GB</th>
    </tr>
    <tr align="center">
        <th>FGVC-Aircraft</th><th>100</th><th>6667</th><th>3333</th><th><a href="https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/">https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/</a></th><th>2.7GB</th>
    </tr>
    <tr align="center">
        <th>Stanford-Dogs</th><th>196</th><th>9144</th><th>8041</th><th><a href="http://vision.stanford.edu/aditya86/ImageNetDogs/">http://vision.stanford.edu/aditya86/ImageNetDogs/</a></th><th>0.7GB</th>
    </tr>
    <tr align="center">
        <th>Stanford-Cars</th><th>120</th><th>12000</th><th>8580</th><th><a href="http://ai.stanford.edu/~jkrause/cars/car_dataset.html">http://ai.stanford.edu/~jkrause/cars/car_dataset.html</a></th><th>1.8GB</th>
    </tr>
    <tr align="center">
        <th>iNaturalist 2017</th><th>5089</th><th>579184</th><th>95986</th><th><a href="https://tensorflow.google.cn/datasets/catalog/i_naturalist2017">https://tensorflow.google.cn/datasets/catalog/i_naturalist2017</a></th><th>238.1GB</th>
    </tr>
</table>
If the official addresses of the above four public datasets are not accessible, they can be easily obtained from my online storage address <a href="https://pan.baidu.com/s/1A6y98oM1kTIFE5Xn2Qvqqg">link_address</a>  with the password <a href="#">1feg</a>.

## Applications

### 🌅 Fine-grained Visual Categorization Tasks

## Released Models
Limited by the experimental conditions, the large number of parameters model was not tested in this network.
<table border="1" width="90%">
    <tr align="center">
        <th>Name</th><th>Pretrain</th><th>acc@1 on CUB</th><th>#param</th><th>Download</th>
    </tr>
    <tr align="center">
        <th>MAM_Base</th><th>ImageNet-21K</th><th>90.9</th><th>86.4m</th><th><a href="https://pan.baidu.com/s/1jRQ_xBfX76gYMEvlBTuf6Q">model_mam.pth</a></th>
    </tr>
    <tr align="center">
        <th>MAM&RSM_Base</th><th>ImageNet-21K</th><th>91.3</th><th>86.4m</th><th><a href="https://pan.baidu.com/s/1jRQ_xBfX76gYMEvlBTuf6Q">model_mam_rsm.pth</a></th>
    </tr>
    <tr align="center">
        <th>AAM_Base</th><th>ImageNet-21K</th><th>90.7</th><th>86.4m</th><th><a href="https://pan.baidu.com/s/1jRQ_xBfX76gYMEvlBTuf6Q">model_aam.pth</a></th>
    </tr>
    <tr align="center">
        <th>MARS-Trans_Base</th><th>ImageNet-21K</th><th>91.9</th><th>86.4m</th><th><a href="https://pan.baidu.com/s/1jRQ_xBfX76gYMEvlBTuf6Q">model_mars.pth</a></th>
    </tr>
</table>
