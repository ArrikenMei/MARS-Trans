# MARS-Trans: Multistage Attention Region Supplement Transformer for Fine-grained Visual Categorization

An improved model with Vision Transformer as the backbone called MARS-Trans is now used in Fine-grained Visual Categorization and has achieved competitive results across multiple FGVC public datasets. In particular, SOTA was achieved on the Stanford Cars dataset.

## Introduction
"MARS-Trans" is a variant model of the Vision Transformer proposed by the Big Data Analysis Lab of the School of Information Engineering, Henan University of Science and Technology.
<div align=left>
<img src='./docs/Figure2.png' width=900>
</div>

## Applications

### 🌅 Fine-grained Visual Categorization Tasks

"MARS-Trans" achieved competitive results on multiple FGVC public datasets and State-of-The-Art results on Stanford-Cars datasets. Performed well on the large dataset iNaturalist 2017, second only to the SIM-Trans model.

**Performance**
 - "MARS-Trans" is mainly improved in three parts: MAM,RSM and AAM. The following table shows the impact of each part on model performance.
<table border="1" width="90%">
	<tr align="center">
        <th colspan="3"> Module</th><th colspan="4"> DataSet/Accuracy(%) </th>
    </tr>
    <tr align="center">
        <th>MAM</th><th>RSM</th><th>AAM</th><th>CUB200-2011</th><th>Stanford-Cars</th><th>Stanford-Dogs</th><th>iNaturalist 2017</th>
    </tr>
    <tr align="center">
        <th> </th><th> </th><th> </th><th>90.2</th><th>93.5</th><th>91.2</th><th>67.0</th>
    </tr>
    <tr align="center">
        <th>√</th><th> </th><th> </th><th>90.9</th><th>94.2</th><th>91.8</th><th>68.1</th>
    </tr>
    <tr align="center">
        <th>√</th><th>√</th><th> </th><th>91.3</th><th>94.7</th><th>92.1</th><th>68.9</th>
    </tr>
    <tr align="center">
        <th> </th><th> </th><th>√</th><th>90.7</th><th>94.2</th><th>91.9</th><th>67.6</th>
    </tr>
    <tr align="center">
        <th>√</th><th>√</th><th>√</th><th>91.9</th><th>95.3</th><th>92.4</th><th>69.5</th>
    </tr>
</table>

<br>


## Released Models


