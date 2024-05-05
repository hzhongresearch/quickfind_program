# Description of QuickFind program

This program is called QuickFind. It is designed to quickly segment a depth map and compute features from each segment. The features can be used to perform object detection when passed on to a machine learning algorithm. The depth maps are expected to come from a sensor like the Kinect V1. The program works with depth maps which have each pixel in 16 bit unsigned format. The program will produce: a PNG with colour coded segments, a CSV with number coded segments, a CSV containing features computed from each segment and a human viewable PNG of the depth map.

There is one change from the original research paper. The feature which computes the median value of pixels in a block is replaced with mean for improved speed and code maintenance.

QuickFind was created during my PhD. The research paper "Quickfind: Fast and contact-free object detection using a depth sensor" was presented at PerCom Workshops 2016. My research is in Ubiquitous Computing and Computer Vision. You can send questions to: henry (dot) zhong (at) unswalumni (dot) com . If you use this work please consider citing our paper.

```
@inproceedings{zhong2016quickfind,
  title={Quickfind: Fast and contact-free object detection using a depth sensor},
  author={Zhong, Henry and Kanhere, Salil S and Chou, Chun Tung},
  booktitle={Pervasive Computing and Communication Workshops (PerCom Workshops), 2016 IEEE International Conference on},
  pages={1--6},
  year={2016},
  organization={IEEE}
}
```

The code, data and paper can be downloaded at the following link.

```
https://hzhongresearch.github.io/
```

### License
Copyright 2016 HENRY ZHONG. The code is released under MIT license. See LICENSE.txt for details.

### Usage instructions
This version of the code has been tested under Debian Linux. Before use, first install prerequisite packages:

```
sudo apt update
sudo apt install build-essential cmake git libopencv-dev pkg-config
```

Download and compile.

```
git clone https://github.com/hzhongresearch/quickfind_program.git
cd quickfind_program
cmake .
make
```

Run with included sample test data using the following commands.

```
./QuickFind_exe 1481625872188_raw_grey.png output 5 5 2 0 0 200 0 0 270 0 0 270 0 0 1 1000
```

### Explanantion of parameters
```
./QuickFind_exe <Input depth map> <Output file name> <Horizontal blocks> <Vertical blocks> <n1> <n2> <n3> <d1> <d2> <d3> <w1> <w2> <w3> <h1> <h2> <h3> <s1> <s2>
```

1. ```<Input depth map>``` : location of the input depth images.
2. ```<Output file name>``` : the results of the program will be saved into files pre-pended with this file path and name.
3. ```<Horizontal blocks> <Vertical blocks>``` : corresponds with params n, m in the paper. Each segment is divided into blocks which are (Horizontal blocks) x (Vertical blocks) in resolution. Feature computation is applied to each block. Increase this value if input resolution is higher to improve object detection accuracy.
4. ```<n1> <n2> <n3>``` : corresponds with params n1, n2 in the paper. Controls maximum absolute difference threshold allowed between neighbouring pixels. Larger n1, n2 will increase threshold exponentially. For now n3 is unused.
5. ```<d1> <d2> <d3>``` : corresponds with params d1, d2 in the paper. Controls maximum depth threshold of segment. Larger d1, d2 will increase threshold linearly. For now d3 is unused.
6. ```<w1> <w2> <w3>``` : corresponds with params w1, w2 in the paper. Controls maximum width threshold of segment. Larger w1, w2 will increase threshold linearly. For now w3 is unused.
7. ```<h1> <h2> <h3>``` : corresponds with params h1, h2 in the paper. Controls maximum height threshold of segment. Larger h1, h2 will increase threshold linearly. For now h3 is unused.
8. ```<s1> <s2>``` : corresponds with feature scaling params in the paper. Block values which are non-zero are scaled between s1 s2.
