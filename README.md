# Hybrid-GCNN-CNN
This is an implementation of Hybrid-GCNN-CNN on PyTorch Geometric coupled with the official YOLACT implementation. The model based on CNN graphs utilizes outputs of the YOLACT object detection model to classify scene between indoor and outdoor. Our approach takes advantage of semantic and spatial information to better characterize the contextual information of the scene. This model can be easily added to any object detection/segmentation models as an add-on.

The repository includes:
-  Source code of Hybrid-GCNN-CNN models
-  Implementations for training and inference
-  Weights for each architecture models
-  Jupyter notebooks version
-  Annotated CD-COCO dataset for training and evaluation
-  Evaluation on CD-COCO dataset

# Installation
To install the YOLACT framework follow the instructions described on the official page: https://github.com/dbolya/yolact

To install the Hybrid-GCNN-CNN framework, follow the instruction below:
      ```
pip install -r requirements.txt
      ```

