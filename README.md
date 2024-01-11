# low-code-opencv

### Introduction 
This code is metioned in the conference paper  ECTI-CON 2024
So, the concept of this low-code based OpenCV in Fig below 

![alt text](https://github.com/PudPawat/low-code-opencv/blob/main/images/concept.png?raw=true)

### Installation 
using python 3.8 
```
pip install requirement.txt
```

### How to use 

#### go to config/ and edite main.json

```
{
  "basic": {
    "source": "dataset/20230318",
    "resize": "False",
    "camera_config": "config/camera_config/a2A2590-60ucPRO_40065215.pfs",
    "process_name": ["erode","sharp","blur","thresh","line","HSV","dilate","canny","circle","sobel","barrel_distort","crop","contour_area"],
    "process": ["crop","barrel_distort","blur","HSV","erode","dilate","contour_area"],
    "config_path": "./config/"
  }
}
```

source is the images folder which want to use this low-code
process_name contains all the image processing functions which cam be used by this low-code platform 
process is a list contain the functions you want to use in the platform respectively.

Then run the code
```
python setting_params.py
```

after the code is runing, there are window to set parameters following the process in the configuration file  main.json

example 
![alt text](https://github.com/PudPawat/low-code-opencv/blob/main/images/windows.png?raw=true)


#### command in UI \
Press A to update to parameters\ 
Press S to save\
Press N to Next the image \
Press R to previous image \
Press Q to quit

#### Pameters file 
the file will be params.json 
```
{
    "HSV": [
        0,
        0,
        55,
        180,
        255,
        255
    ],
    "gaussianblur": [
        1,
        1
    ],
    "dilate": [
        5,
        0
    ],
    "erode": [
        15,
        0
    ],
    "canny": [
        10,
        10
    ],
    "circle": [
        1,
        5,
        0,
        20
    ]
}
```

### SDK usage 

```
from lib_save import Imageprocessing, read_save

params = {'HSV': [..]} ## looks like above example
reading = read_save()
img_params,_,_ = reading.read_params(params, frame_ori)
```

## Tutorial for more
then go to setting up the parameters 
I suggest to visit this/

![Watch the video](https://drive.google.com/file/d/14nNEcNqJ4s8NHQkEZMCcJOjjBIxKbBbl/view?usp=sharing)
https://drive.google.com/file/d/14nNEcNqJ4s8NHQkEZMCcJOjjBIxKbBbl/view?usp=sharing

see more details https://docs.google.com/document/d/1szQmotkCh7ohC_64nXKTcr0Ttbz6_SV4jJaD74pe12c/edit?usp=sharing





