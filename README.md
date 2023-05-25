# ADS Wildfire: Team Creek

CONTENTS
--------

<!-- TOC -->
 * [Introduction](#introduction)
 * [Installation](#installation)
 * [User Instructions](#user-instructions)
 * [Contributers](#contributers)
 * [References](#references)
 * [Licensing](#licensing)
<!-- TOC -->
 

INTRODUCTION
------------

This repository contains PyTorch models, and example code for running them, to forecast wildfire growth. The models were trained on satellite imagery from the 2018 Ferguson wildfire.

The documentation of the package, with a detailed explanation of all classes and functions, can be accessed [here](docs/html/index.html). In there, you can also find a short description explaining the forecasting models and algorithms used.

INSTALLATION
------------

To install pre-requisites, from the base directory run:
```
pip install -r requirements.txt
pip install -e .
```  

User Instructions
------------

Please place downloaded data under `wildfire/models/data`.

The saved weights for our trained models have been included under `wildfire/models/weights`. These weights are loaded in the respective notebooks. If wanting to retrain the models from scratch, please delete/remove the saved weights from the above directory. In this case, the `objective_1` and `objective_2` notebooks must be run before the `objective_3` notebook as this requires the saved weights. Warning: our models take quite a long to train from scratch, so training on Google Colab is recommended. 
 
Contributers
------------

* [Josh Millar](mailto:joshua.millar22@imperial.ac.uk)
* [Keyi Zhu](mailto:keyi.zhu22@imperial.ac.uk)
* [Hang, Zhao](mailto:hang.zhao22@imperial.ac.uk)
* [Yicheng Wang](mailto:yicheng.wang22@imperial.ac.uk)
* [Sitong Mu](mailto:sitong.mu22@imperial.ac.uk)
* [Yu Yan](mailto:yu.yan22@imperial.ac.uk)
* [Chaofan Wu](mailto:chaofan.wu22@imperial.ac.uk)
* [Elena Mustafa](mailto:elena.mustafa22@imperial.ac.uk)


If you have any usage questions or improvement recommendations, feel free to contact us!

References
------------
* Shi, X. et al. (2015) Convolutional LSTM network: A machine learning approach for precipitation nowcasting, arXiv.org. Available at: https://arxiv.org/abs/1506.04214 (Accessed: 24 May 2023). 

LICENSING
------------
MIT License

Copyright (c) [2023] [Team Creek]

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
