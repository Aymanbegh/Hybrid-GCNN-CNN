# Hybrid-GCNN-CNN
This is an implementation of Hybrid-GCNN-CNN on PyTorch Geometric coupled with the official YOLACT implementation. The model based on CNN graphs utilizes outputs of the YOLACT object detection model to classify scene between indoor and outdoor. Our approach takes advantage of semantic and spatial information to better characterize the contextual information of the scene. This model can be easily added to any object detection/segmentation models as an add-on.

The repository includes:
-  Source code of Hybrid-GCNN-CNN models
-  Implementations for training and inference
-  Weights for each architecture models
-  Jupyter notebooks version
-  Annotated CD-COCO dataset for training and evaluation
-  Evaluation on CD-COCO dataset

# Requirements

Python 3.8, PyTorch = torch 2.0.1, torch-geometric-2.5.3 and other common packages listed in requirements.txt.

**CD-COCO Requirements**:
To train or test on CD-COCO, you'll also need:
- pycocotools (installation instructions below)
- CD-COCO Dataset (images, object annotations and scene type groundthruth): https://github.com/Aymanbegh/CD-COCO
- Scene type groundthruth is contained into the trainings.txt file


# Installation
Download the Hybrid-GCNN-CNN repository and enter it:

```
git clone https://github.com/Aymanbegh/Hybrid-GCNN-CNN.git
cd Hybrid-GCNN-CNN
```

To install the YOLACT framework follow the instructions described on the official page: https://github.com/dbolya/yolact
Clone its repository into the Hybrid-GCNN-CNN folder and enter it:

The tree should look like this:

      ```
      ├── Hybrid-GCNN-CNN
          └── requirements.txt
          └── install.py
          └── inference.py
          └── evaluate.py
          └── train.py
          └── gin.py
          └── gcn.py
          └── ginlaf.py
          └── weights
          └── data
                └── train.json
                └── trainings.txt
                └── train2017_distorted
                └── train2017
          └── yolact
            └── yolact tree...              
         ```  

To install the Hybrid-GCNN-CNN framework, follow the instruction below:
      ```
pip install -r requirements.txt
      ```

To download the Hybrid-GCNN-CNN data and weights, follow the instruction below:
      ```
python installation.py {instruction}
      ```

where instruction is one of the following command:
- **all_distorted**: download the Hybrid-GCNN-CNN weights, the yolact wieghts, train.json, the training cd-coco distorted images
- **all_normal**: download the Hybrid-GCNN-CNN weights, the yolact weights, train.json, the train ms-coco original images
- **weight**: download the Hybrid-GCNN-CNN weights, the yolact weights


# Method


Model wieghts:

|Model| Hidden layer size  | Inference speed (ms) | Precision | link | 
| ------ | :------: | :------: | :------: | :------: |  :------: |
| **GCN** | 1024 | 0.470 | 88.9% | xxx |
| **GIN** | 1024 | 0.123 | 90.6% | xxx |
| **GINLAF** | 32 | 0.110 | 92.0% | xxx |


# Instructions for use

# Citation
**CD-COCO: A versatile complex distorted coco database for scene-context-aware computer vision**
```
@inproceedings{beghdadi2023cd,
  title={Cd-coco: A versatile complex distorted coco database for scene-context-aware computer vision},
  author={Beghdadi, Ayman and Beghdadi, Azeddine and Mallem, Malik and Beji, Lotfi and Cheikh, Faouzi Alaya},
  booktitle={2023 11th European Workshop on Visual Information Processing (EUVIP)},
  pages={1--6},
  year={2023},
  organization={IEEE}
}
```

**Hybrid-GCNN-CNN: A New Lightweight Hybrid Graph Convolutional Neural Network--CNN Scheme for Scene Classification using Object Detection Inference**

```
@article{beghdadi2024new,
  title={A New Lightweight Hybrid Graph Convolutional Neural Network--CNN Scheme for Scene Classification using Object Detection Inference},
  author={Beghdadi, Ayman and Beghdadi, Azeddine and Ullah, Mohib and Cheikh, Faouzi Alaya and Mallem, Malik},
  journal={arXiv preprint arXiv:2407.14658},
  year={2024}
}
```
