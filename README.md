# QuickFind program
QuickFind is a fast segmentation and object detection algorithm using only depth maps. Depth maps are images captured from depth sensors like Kinect. The idea is in the future depth sensors will be common so such an algorithm will be useful. This project was created during my PhD. The associated research paper was presented at PerCom Workshops 2016.

There is one change from the original research paper. The feature which computes the median value of pixels in a block is replaced with mean for improved speed and code maintenance.

This is the code for QuickFind.

## Instructions
This version of the code has been tested under Debian Linux. Before use, first install prerequisite packages:

```
sudo apt update
sudo apt install build-essential cmake git libopencv-dev
```

Download and compile.

```
git clone https://github.com/hzhongresearch/quickfind_program.git
cd quickfind_program
cmake .
make
```

Run with included sample test data using the following command.

```
./QuickFind_exe 1481625872188_raw_grey.png output 5 5 2 0 0 200 0 0 270 0 0 270 0 0 1 1000
```

## Explanantion of parameters
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

# Links

* [Website](https://hzhongresearch.github.io/)
* [Paper](https://www.researchgate.net/publication/301583832_QuickFind_Fast_and_contact-free_object_detection_using_a_depth_sensor)
* [Code](https://github.com/hzhongresearch/quickfind_program)
* [Data](https://huggingface.co/datasets/hzhongresearch/quickfind_mask_data)

# Licence
Copyright 2016 HENRY ZHONG. The code is released under MIT licence. See [LICENCE.txt](LICENCE.txt).

If you use this work please cite our paper.

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
