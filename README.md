# Salmon respiration frequency evaluation by deep learning
This repository contains code for automated individual salmon frequency extraction from video sequences. It was developed during the spring of 2023 as part of my master thesis at NTNU. Some notes follow below.


- Data and trained models are not included in this repository.
- The code for training Keypoint RCNN and Resnet models are not included, as the network training code were written and ran on the kaggle.com plattform.
- Prewritten scripts from the pytorch/vision github page: https://github.com/pytorch/vision/tree/main/references/detection were used when training the neural networks.
- Labelme was used for manual annotation: https://github.com/wkentaro/labelme/releases

Below, a video visualizing the output of the salmon frequnecy extraction pipeline up until the sine wave function fitting (see the thesis report) is displayed.

https://github.com/espenbh/salmon_welfare_analysis_by_automated_respiration_frequency_extraction/assets/59967194/b4101865-f772-48d9-a2cc-14af3106b138

