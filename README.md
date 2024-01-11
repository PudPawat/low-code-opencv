# low-code-opencv

### Introduction 
This code is metioned in the conference paper  ECTI-CON 2024
So, the concept of this low-code based OpenCV in Fig below 

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



