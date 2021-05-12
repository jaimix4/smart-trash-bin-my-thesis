# smart-trash-bin-my-thesis

This repository contains samples of media, code, and dataset from my thesis: 

**`AUTOMATIC WASTE SORTING EXPERIMENTATION USING A NOVEL SMART TRASH BIN WITH CNN`**

By the time of this commit, the main purpose of this repository is to showcase my work for my master's applications. Dear admission committee I kindly ask to not share this repository, since a patent related to the robot is in process. Later on, more information that will be permanently available and for public use, will be published. Thanks in advance!

Summary in standard format: [thesis_summary.pdf](https://github.com/jaimix4/smart-trash-bin-my-thesis/files/6422415/thesis_summary.pdf)

## Introduction 

Recycling has proven to be a partial solution to the global problem related to the accumulation of waste in the last decades. Nonetheless, overall recycling efficiency is reduced due the lack of an automatic classification method in large scale recycling facilities and on site waste collection. The current work offers a novel smart trash device capable of automatically classifying and segregating common recyclables objects on site. 

<p align="center">
<img align="center" width="500" src="https://user-images.githubusercontent.com/31749600/117710978-54283a80-b198-11eb-87b8-558f49430365.jpg">
</p>

The automatic classification is done with a convolutional neural network (CNN) model, computer vision algorithm and a common RGB camera for input. Once an object is thrown in the device, it is classified, then with the use of servo motors in a clever mechanical system, the object is physically segregated in the designated compartments. The next section will explain in more detail each part of the overall system.

## Fotini10k dataset

The most important factor for the performance of CNNs is the quality of the dataset it is trained on. Quality in this sense can be translated into: number of examples, whether these examples are representative of the data the CNN will encounter in the "wild". The dataset used for fine-tuning the CNNs in this work is the Fotini10k dataset. This dataset features images of the following categories of recyclable objects: `plastic bottles`, `aluminum cans` and `paper and cardboard`. Examples can be found in the figures below. This dataset was developed in a previous work. More information about this dataset can be found here [[1]](#references).

<p align="center">
<img width="500" src="https://user-images.githubusercontent.com/31749600/117036104-b54c9b80-acca-11eb-9823-75a1e01192db.jpg">
</p>

## CNNs for Image Classification

The CNN used in this work is MobileNetV2 [[2]](#references) tailor for image classification. It was previously trained on the ImageNet dataset and its weights were fine tuned with the previously mentioned [Fotini10k dataset](#fotini10k-dataset) in cropped form. This CNN was chosen, because [[1]](#references) concluded to be more suited to the purpose of this work. The modifications done to the architecture of MobileNetV2 and the training metrics are presented in the figures below.

<p align="center">
<img width="500" src="https://user-images.githubusercontent.com/31749600/117040759-d663bb00-accf-11eb-8262-3fceb1a01239.png">
</p>

> Modifications to the architecture of MobileNetV2

<p align="center">
<img width="500" src="https://user-images.githubusercontent.com/31749600/117040824-eb404e80-accf-11eb-81e6-4e6df8dfc98f.png">
</p>

> Training log of MobileNetV2 while fine-tuning with the Fotini10k dataset

The CNN then was tested on a test set. It scored a top-1 accuracy of 99.3%. Then, it was quantized to `int8` weights to be used on a Raspberry Pi 4 along a [Coral USB accelerator](https://coral.ai/products/accelerator/). The quantization was done according to the [post-training quantization guidelines](https://www.tensorflow.org/lite/performance/post_training_quantization) from tensorflow. With quantization the top-1 accuracy dropped to 96.7%, which is more than useful.

## Object Detection Approach

For accurate classification of the objects, they first need to be localized by a computer vision algorithm that extracts a region of interest (ROI) that is inputted to the CNN. The device features an enclosure where the object is deposited and then the camera takes frames for the classification. 

<p align="center">
<img width="500" src="https://user-images.githubusercontent.com/31749600/118022079-92049a80-b321-11eb-81d6-28865d2acaec.jpg">
</p>

This enclosure has a constant background, for this reason the computer vision algorithm is based on a background subtraction (BS) with a mixture of Gaussian model (MOG). This BS model detects the presence of an object and then, with a saliency algorithm, the ROI is extracted and then passed to the CNN. It is basically a two-step object detector. A diagram of this process is presented below.

[Diagram CV]

Examples of how the algorithm works are presented below.




<p align="center">
<img width="500" src="https://user-images.githubusercontent.com/31749600/118025407-6d122680-b325-11eb-9327-0390de073ca6.png">
<img width="500" src="https://user-images.githubusercontent.com/31749600/118025415-713e4400-b325-11eb-9d4e-4e15a49bb147.png">
</p>


A simple code of this methodology is presented in this [collab notebook](https://colab.research.google.com/notebooks/intro.ipynb).

Overall, the classification system is a two stage object detector which features an % mAP at 75% IoU on the test set. The system is capable of running at 30 FPS at presence of object detection mode and 7 FPS at inference mode. These metrics were taken while running the algorithm in the Raspberry Pi 4 with Coral USB accelerator.

### End-to-End Object Detection Alternative

There is also the option of using an end-to-end object detection CNN. This option was explored but expenses related to cloud computing moved the exploration of this approach to further research. The limited test done with this technique indicates that SSD-MobileNetV2 or SSDLite-MobileDet, seems suitable for the application. After the thesis work, a paper related performance, cost, time comparisons of two step objects detectors vs end-to-end object detectors for large deployment applications is scheduled for submission.

## Smart Bin Design

The physical segregation is done with a mechanical system mainly based on RC servos actuators. It was designed with 3D CAD modeling software. The figures below show some features of the design.

[Images ...]

## Dynamical Simulations for Electromechanical System Validation

The kinematics and dynamics were validated with MATLAB Simscape Multibody. The control system was validated with 3D dynamic simulations, to mainly estimate the power consumption of the device. Overall, The system runs at 5 V power supply, in standby operation the system consumes < 4 A and during segregation a maximum of 9 A validated with simulations and physical experimentation. The figures below show the block diagram of the system and media of some simulations.

[block diagram .... simulation of main actuator with current graph to the side]

## Integration of Software and Hardware

Information yet to be released. Final tests and writing

## References

[1] 

[2] 


