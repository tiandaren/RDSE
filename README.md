# Road damage detection and calculation

## Prerequisites

You need to install:

- [Python3 >= 3.6](https://www.python.org/downloads/)

- Use `requirements.txt` to install required python dependencies

  ```
  # Python >= 3.6 is needed
  pip3 install -r requirements.txt
  ```

## Detection 

1. Go to `yolov5` directory

   ```
   cd yolov5
   ```

2. Execute follwoing commands 

   ```
   python detect.py --weights yolo_weights/l1.pt yolo_weights/s2.pt yolo_weights/s1.pt --classes 0 2 4 5 --augment
   ```

   

