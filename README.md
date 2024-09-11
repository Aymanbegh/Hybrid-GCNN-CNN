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

The YOLACT framework is already integrated into the repository. To install the YOLACT ++ framework follow the instructions described on the official page: https://github.com/dbolya/yolact :

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
                └── model2_GCN_X2_1.pth
                └── model2_GIN_X2_1.pth
                └── model2_GINLAF_X2_1.pth
          └── data
                └── train.json
                └── trainings.txt
                └── train2017_distorted
                      └── image 1
                      └── ...
                └── train2017
                      └── image 1
                      └── ...
          └── yolact files...              
         ```  

To install the Hybrid-GCNN-CNN framework, follow the instruction below:
      ```
pip install -r requirements.txt
      ```

To download the Hybrid-GCNN-CNN data, follow the instruction below:
      ```
python preparation.py {instruction}
      ```

where instruction is one of the following command:
- **all_distorted**: download the yolact weights, train.json from the cd-coco dataset, the training cd-coco distorted images -> for training or evaluation
- **all_normal**: download the yolact weights from the cd-coco dataset, train.json, the train ms-coco original images -> for training or evaluation

**Important**: You have to download manually the yolact weights (yolact_im700_54_800000.pth by default) -> for inference, or evaluation


# Method
The overall architecture of our new Lightweight Hybrid Graph Convolutional Neural Network - CNN Scheme (LH-G2CNN) is illustrated in figure below. The proposed model consists
of a CNN object detection model that conveys semantic and spatial information to a GCNN model for boosting the scene classification process.

Model information without yolact inference by directly using groundtruth from the cd-coco dataset (bounding boxes and labels):

|Model| Hidden layer size  | Inference speed (ms) | Precision | 
| ------ | :------: | :------: | :------: | 
| **GCN** | 1024 | 0.470 | 88.9% |
| **GIN** | 1024 | 0.123 | 90.6% | 
| **GINLAF** | 32 | 0.110 | 92.0% | 


# Instructions for use

**Training commands**

Command for training without yolact object detection model:

```
python  train.py --dataset=./data/train.json --file_res="./datat/trainings.txt" --nb_label=1 --model="model_name" --hidden=1024
```

Command for training with yolact object detection model:

```
python  training.py --dataset=./data/train.json --trained_model=./weights/yolact_im700_54_800000.pth --score_threshold=0.15 --top_k=15 --image_dir="./data/train2017_distorted/" --file_res="./datat/trainings.txt" --nb_label=1 --model="model_name" --hidden=1024
```

Where "model_name" is GCN, GIN or GINLAF. You can adjust nb_label and hidden_channel parameters

**Evaluation commands**

Command for evaluation without yolact object detection model:

```
python  evaluate.py --dataset=./data/train.json --file_res="./data/trainings.txt" --gcn_model="./weights/model2_{model_name}_X2_1.pth" --nb_label=1 --model="model_name" --hidden=1024 --object_detection=False
```

Command for training with yolact object detection model:

```
python  evaluate.py --dataset=./data/train.json --trained_model=./weights/yolact_im700_54_800000.pth --score_threshold=0.15 --top_k=15 --image_dir="./data/train2017_distorted/" --file_res="./datat/trainings.txt" --gcn_model="./weights/model2_{model_name}_X2_1.pth" --nb_label=1 --model="model_name" --hidden=1024 --object_detection=True
```

**Inference commands**

Command for training with yolact object detection model:

```
python  inference.py --trained_model=./weights/yolact_im700_54_800000.pth --score_threshold=0.15 --top_k=15 --image_dir="./data/train2017_distorted/"  --gcn_model="./weights/model2_{model_name}_X2_1.pth" --nb_label=1 --model="model_name" --hidden=1024

```

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
