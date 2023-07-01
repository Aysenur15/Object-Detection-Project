# Object-Detection-Project


In this project, I have created an object training model from a dataset which includes images and annotations files. 

Firstly, I have uploaded the my images and annotations to roboflow. It created a custom dataset according to the annotations and augmentations.

**The link of the project in roboflow:** https://universe.roboflow.com/object-detection-xht9p/object-detection-project-v3abc/dataset/1

**Also, I have worked with Google Colab after creating a custom dataset from roboflow :** https://colab.research.google.com/drive/1fp-guefxM-I_Nu67NVYq0Dvb5aAdSXua?usp=sharing

![image](https://github.com/Aysenur15/Object-Detection-Project/assets/100716886/8b7a3166-7382-4a0d-8acc-49f693fd438a)

**INSTALLING YOLOv5**

I have cloned a github repository for YOLOv5 object detector model and set up YOLOv5 to install dependincies. 
```
#clone YOLOv5 and
!git clone https://github.com/ultralytics/yolov5  # clone repo
%cd yolov5
%pip install -qr requirements.txt # install dependencies
%pip install -q roboflow

import torch
import os
from IPython.display import Image, clear_output  # to display images

print(f"Setup complete. Using torch {torch.__version__} ({torch.cuda.get_device_properties(0).name if torch.cuda.is_available() else 'CPU'})")
```

**GETTING DATASET from ROBOFLOW**

To work with the dataset, I got uploded the dataset I have created with roboflow. Then, I also setup the environment that will work with the dataset.
```
from roboflow import Roboflow
!pip install roboflow

from roboflow import Roboflow
rf = Roboflow(api_key="XmXn0BfZxMJpS2RvbDyg")
project = rf.workspace("object-detection-xht9p").project("object-detection-project-v3abc")
dataset = project.version(1).download("yolov5")
```

```
import os
os.environ["DATASET_DIRECTORY"] = "/content/datasets"
```

**TRAINING the YOLOv5 MODEL**

After installations, I started training with the code below:
```
!python train.py --img 640 --batch 16 --epochs 100 --data {dataset.location}/data.yaml --weights yolov5s.pt --cache
```

**TRAINING RESULTS**

To see the training results, I wrote a code to see the tensorboard and I saved logs in the folder "runs".

```
%load_ext tensorboard
%tensorboard --logdir runs
```
![image](https://github.com/Aysenur15/Object-Detection-Project/assets/100716886/af4db766-bc98-43f2-a1c0-9ae2a2ab04ca)


**DETECTION**

The results are saved in the folder "runs". Then, I started detection of the training model with the codes below:
```
!python detect.py --weights runs/train/exp/weights/best.pt --img 640 --conf 0.2 --source {dataset.location}/test/images

!python detect.py --weights runs/train/exp/weights/best.pt --img 640 --conf 0.2 --source {dataset.location}/valid/images

!python detect.py --weights runs/train/exp/weights/best.pt --img 640 --conf 0.2 --source {dataset.location}/train/images
```
**DISPLAYING INFERENCE IMAGES**

After the detection, I needed to display the images that I have tested with the training model. For each detected documents which are train,valid and test, I have displayed the images with the codes:
```
import glob
from IPython.display import Image, display

i = 0

for imageName in glob.glob('/content/yolov5/runs/detect/exp/*.jpg'):
    i += 1

    if i < 48:
      display(Image(filename=imageName))
      print("\n")
```

```
import glob
from IPython.display import Image, display

i = 0

for imageName in glob.glob('/content/yolov5/runs/detect/exp2/*.jpg'):
    i += 1

    if i < 97:
      display(Image(filename=imageName))
      print("\n")
```

```
import glob
from IPython.display import Image, display

i = 0
for imageName in glob.glob('/content/yolov5/runs/detect/exp3/*.jpg'):
    i += 1

    if i < 1008:
      display(Image(filename=imageName))
      print("\n")

```

Then, I have seen the results of the training model on the images.

![image](https://github.com/Aysenur15/Object-Detection-Project/assets/100716886/83ed510f-b2e0-449f-b25b-9ae8ccbc5831)
![image](https://github.com/Aysenur15/Object-Detection-Project/assets/100716886/67171df8-8802-4ef8-9622-2375835313e2)


You can see the other images and whole project from the Google Colab link above.
