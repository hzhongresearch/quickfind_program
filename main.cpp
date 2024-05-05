#include <string>
#include <vector>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <opencv2/opencv.hpp>

#include "quickfind.h"

using namespace std;
using namespace std::chrono;
using namespace cv;

int main(int argc, char **argv)
{
    if (argc != 19)
    {
        cout << "Usage: \n./QuickFind_exe <Input depth map> <Output file name> <Horizontal blocks> <Vertical blocks> ";
        cout << "<n1> <n2> <n3> <d1> <d2> <d3> <w1> <w2> <w3> <h1> <h2> <h3> <s1> <s2>" << endl;
        return EXIT_SUCCESS;
    }

    // Input file path, gets depth map using this file path
    string input_file_path = argv[1];
    // Output file name, all output files will have this string prepended
    string output_scene_name = argv[2];

    // Divide each segment into block_across X block_down for feature computation
    int block_across = stoi(argv[3]);
    int block_down = stoi(argv[4]);

    // Set neighbouring difference parameter
    vector<double> ndiff { stod(argv[5]), stod(argv[6]), stod(argv[7]) };
    // Set depth parameter
    vector<double> depth { stod(argv[8]), stod(argv[9]), stod(argv[10]) };
    // Set width parameter
    vector<double> width { stod(argv[11]), stod(argv[12]), stod(argv[13]) };
    // Set height parameter
    vector<double> height { stod(argv[14]), stod(argv[15]), stod(argv[16]) };

    // Scaling parameters
    double scale_start = stod(argv[17]);
    double scale_end = stod(argv[18]);

    // Read a depth map.
    // The depth map can be from RGBD dataset. Dataset can be downloaded from:
    // http://rgbd-dataset.cs.washington.edu/
    // Tested with data from Kinect for Xbox 360: 640X480 16 bit unsigned depth maps.
    // Each pixel in PNG should correspond to a depth map cell.
    // Each cell is expected to be 16bit unsigned, representing distance from sensor.
    Mat depth_cells = imread(input_file_path, IMREAD_ANYCOLOR | IMREAD_ANYDEPTH);

    // Start timer
    high_resolution_clock::time_point t1 = high_resolution_clock::now();

    // Run segmentation algorithm and feature extraction
    Processed_depth_map process_depth = Algorithms::Segmentation(depth_cells, block_across, block_down, width, height, depth, ndiff, scale_start, scale_end);

    // End timer
    high_resolution_clock::time_point t2 = high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();
    // Print out execution time in microseconds
    cout << "Time: " << duration << " microseconds" << endl;

    // Print features and images
    Print_results::print_segments_png(process_depth, output_scene_name + "_segments.png");
    Print_results::print_segments_csv(process_depth, output_scene_name + "_segments.csv");
    Print_results::print_features_csv(process_depth, output_scene_name + "_features.csv");
    Print_results::print_viewable_png(process_depth, output_scene_name + "_viewable.png");

    return EXIT_SUCCESS;
}
