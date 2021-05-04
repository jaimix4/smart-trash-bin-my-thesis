# smart-trash-bin-my-thesis

This repository contains samples of media, code, and dataset from my thesis: 

**`AUTOMATIC WASTE SORTING EXPERIMENTATION USING A NOVEL SMART TRASH BIN WITH CNN`**

By the time of this commit, the main purpose of this repository is to showcase my work for my master's applications. Dear admission committee I kindly ask to not share this repository, since a patent related to the robot is in process. Later on, more information that will be permanently available for public use, will be published. Thanks in advance!

Summary in standard format: [thesis_summary.pdf](https://github.com/jaimix4/smart-trash-bin-my-thesis/files/6422415/thesis_summary.pdf)

## Introduction 

Recycling has proven to be a partial solution to the global problem related to the accumulation of waste in the last decades. Nonetheless, overall recycling efficiency is reduced due the lack of an automatic classification method in large scale recycling facilities and on site waste collection. The current work offers a novel smart trash device capable of automatically classifying and segregating common recyclables objects on site. 

![bins_mess](https://user-images.githubusercontent.com/31749600/117036851-7bc86000-accb-11eb-96fa-bb76af48269d.jpg)

The automatic classification is done with a convolutional neural network (CNN) model, computer vision algorithm and a common RGB camera for input. Once an object is thrown in the device, it is classified, then with the use of servo motors in a clever mechanical system, the object is physically segregated in the designated compartments. The next section will explain in more detail each part of the overall system.

## Fotini10k dataset

The most important factor for the performance of CNNs is the quality of the dataset it is trained on. Quality in this sense can be translated into: number of examples, whether these examples are representative of the data the CNN will encounter in the "wild". The dataset used for fine-tuning the CNNs in this work is the Fotini10k dataset. This dataset features images of the following categories of recyclable objects: **plastic bottles**, **aluminum cans** and **paper and cardboard**. Examples can be found in the figures below. This dataset was developed in a previous work. More information about this dataset can be found [here](https://arxiv.org/abs/2104.00868).

![figure_1](https://user-images.githubusercontent.com/31749600/117036104-b54c9b80-acca-11eb-9823-75a1e01192db.jpg)

## CNNs for Image Classification


## Object Detection Approach


### End-to-End Object Detection Alternative


## Smart Bin Design


## Dynamical Simulations for Electromechanical System Validation


## Integration of Software and Hardware


## References

