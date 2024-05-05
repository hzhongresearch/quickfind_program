#ifndef QUICKFIND_H
#define QUICKFIND_H

#include <iostream>
#include <fstream>
#include <tuple>
#include <vector>
#include <deque>
#include <algorithm>
#include <cmath>
#include <limits>

#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

/**
 * @brief The Segment class
 * Stores all information of a single segment of a depth map generated during segmentation
 */
class Segment
{
friend class Processed_depth_map;
friend class Algorithms;
friend class Print_results;
public:
    /**
     * @brief Segment
     * @param depth_map Depth map of the scene
     * @param group_pixels Map id of each segment
     * @param group_id Unique numerical identifier
     * @param block_across Number of block to compute in horizontal axis
     * @param block_down Number of block to compute in vertical axis
     * @param start The starting depth pixel
     * @param scale_start
     * @param scale_end
     */
    Segment(Mat& depth_map, Mat& group_pixels, int group_id, int block_across, int block_down, tuple<int, int, unsigned short> start, double scale_start, double scale_end);

protected:
    /**
     * @brief add Add a new pixel to the segment
     * @param added_pixel
     */
    void add(tuple<int, int, unsigned short> added_pixel);

    /**
     * @brief finalise Once no more pixels need to be added, then produce depth map and features.
     */
    void finalise();

    /**
     * @brief Should_finalise Determine if segment meets min size requirements and consist of more than zero valued pixels
     * @return If return false then invalid segment
     */
    bool Should_finalise();

    /**
     * @brief produce_segment_map
     * Generate the array/Mat which will be used to create bitmap representation of segment
     */
    void produce_segment_map();

protected:
    // Used to compute features
    vector<double> features;
    // Depth map of the scene
    Mat depth_map;
    // Map id of each segment
    Mat group_pixels;
    // Determines if segment is valid, if false after running finalise discard this segment
    bool process;
    // Used to generate bitmap representation of segment
    Mat segment_map;

    // Used to compute features
    tuple<int, int, unsigned short> min_x;
    tuple<int, int, unsigned short> max_x;
    tuple<int, int, unsigned short> min_y;
    tuple<int, int, unsigned short> max_y;
    tuple<int, int, unsigned short> max_depth;
    tuple<int, int, unsigned short> min_depth;

    // Checks if any pixels are non zero
    bool contains_valid_pixels;
    // Store feature parameters and id of segment
    int group_id, block_across, block_down;
    // Record the number of pixels
    size_t pixel_count;
    // Scale feature values between these values
    double scale_start, scale_end;
};

/**
 * @brief The Processed_depth_map class
 * Stores all information of a processed depth map
 */
class Processed_depth_map
{
friend class Algorithms;
friend class Print_results;
public:
    /**
     * @brief Processed_depth_map
     * @param depth_map The raw depth map data
     * @param block_across Block resolution in horizontal axis
     * @param block_down Block resolution in vertical axis
     * @param width Parameters controlling segment width
     * @param height Parameters controlling segment height
     * @param depth Parameters controlling segment growth in axis perpendicular to sensor plane
     * @param ndiff Parameters controlling difference of neighbouring pixels
     * @param scale_start Scale feature values between scale_start scale_end
     * @param scale_end Scale feature values between scale_start scale_end
     */
    Processed_depth_map(Mat& depth_map, int block_across, int block_down, vector<double>& width, vector<double>& height, vector<double>& depth, vector<double>& ndiff, double scale_start, double scale_end);

protected:
    /**
     * @brief add_segment
     * @param new_segment The new segment to be added
     */
    void add_segment(Segment& new_segment);

protected:
    // Read in as parameters
    Mat depth_map;
    vector<double> ndiff, depth, width, height;
    int block_across, block_down;
    double scale_start, scale_end;
    // Created as part of the segmentation
    vector<Segment> segments;
    Mat group_pixels;
    int group_id;
};

/**
 * @brief The Algorithms class
 * Contains the segmentation and feature extraction algorithm
 */
class Algorithms
{
friend class Processed_depth_map;
friend class Segment;
public:
    /**
     * @brief Segmentation Iterate over the depth map and run segmentation algorithm
     * @param depth_pixels The raw depth map data
     * @param block_across Number of block to compute in horizontal axis
     * @param block_down Number of block to compute in vertical axis
     * @param width Parameters controlling segment width
     * @param height Parameters controlling segment height
     * @param depth Parameters controlling segment growth in axis perpendicular to sensor plane
     * @param ndiff Parameters controlling difference of neighbouring pixels
     * @param scale_start Scale feature values between scale_start scale_end
     * @param scale_end Scale feature values between scale_start scale_end
     * @return A Class representing a processed depth map divided into segments
     */
    static Processed_depth_map Segmentation(Mat& depth_pixels, int block_across, int block_down, vector<double>& width, vector<double>& height, vector<double>& depth, vector<double>& ndiff, double scale_start, double scale_end);

protected:
    /**
     * @brief Compute_dimensions Save features representing feature dimensions
     * @param dimensions The computed features values are saved into this vector
     * @param pixel_count Number of non-zero pixels in segment
     * @param min_x Position of left most pixel
     * @param max_x Position of right most pixel
     * @param min_y Position of top most pixel
     * @param max_y Position of bottom most pixel
     * @param min_depth Value of pixel with min depth
     * @param max_depth Value of pixel with max depth
     */
    static void Compute_dimensions(vector<double>& dimensions, unsigned long pixel_count, int min_x, int max_x, int min_y, int max_y, unsigned short min_depth, unsigned short max_depth);
    /**
     * @brief Compute_blocks Compute features by dividing a segment into blocks and computing the mean value of each block
     * @param blocks The computed features values are saved into this vector
     * @param segment_map The mini depth map representing just this segment
     * @param block_across Resolution of block in horizontal axis
     * @param block_down Resolution of block in vertical axis
     * @param min_x Position of left most pixel
     * @param max_x Position of right most pixel
     * @param min_y Position of top most pixel
     * @param max_y Position of bottom most pixel
     * @param min_depth Value of pixel with min depth
     * @param max_depth Value of pixel with max depth
     * @param scale_start Non-zero features values are scaled between scale_start scale_end
     * @param scale_end Non-zero features values are scaled between scale_start scale_end
     */
    static void Compute_blocks(vector<double>& blocks, Mat segment_map, int block_across, int block_down, int min_x, int max_x, int min_y, int max_y, unsigned short min_depth, unsigned short max_depth, double scale_start, double scale_end);
    /**
     * @brief Connect_component_mod Modified connected components algorithm to create a new segment given a starting location
     * @param depth_pixels The raw depth map data
     * @param group_pixels An array representing the pixels already assigned a segment
     * @param start Starting depth pixel
     * @param block_across Resolution of block in horizontal axis
     * @param block_down Resolution of block in vertical axis
     * @param group_id Unique ID for this segment
     * @param width Max width threshold
     * @param height Max height threshold
     * @param depth Max depth threshold
     * @param ndiff Max absolute difference between neighbours
     * @param scale_start Non-zero features values are scaled between scale_start scale_end
     * @param scale_end Non-zero features values are scaled between scale_start scale_end
     * @return A segment of the depth map
     */
    static Segment Connect_component_mod(Mat& depth_pixels, Mat& group_pixels, tuple<int, int, unsigned short> start, int block_across, int block_down, int group_id, vector<double> width, vector<double> height, vector<double> depth, vector<double> ndiff, double scale_start, double scale_end);
    /**
     * @brief compute_max_width Given starting depth pixel compute the max width allowed for segment
     * @param width
     * @param start_depth
     * @return
     */
    static double compute_max_width(vector<double>& width, unsigned short start_depth);
    /**
     * @brief compute_max_height Given starting depth pixel compute the max height allowed for segment
     * @param height
     * @param start_depth
     * @return
     */
    static double compute_max_height(vector<double>& height, unsigned short start_depth);
    /**
     * @brief compute_max_depth Given starting depth pixel compute the max depth allowed for segment
     * @param depth
     * @param start_depth
     * @return
     */
    static double compute_max_depth(vector<double>& depth, unsigned short start_depth);
    /**
     * @brief compute_max_ndiff Given starting depth pixel compute the max absolute difference allowed for neighbouring pixels
     * @param ndiff
     * @param start_depth
     * @return
     */
    static double compute_max_ndiff(vector<double>& ndiff, unsigned short start_depth);
    /**
     * @brief rule_1_inbounds Check if pixel within image boundary
     * @param group_pixels
     * @param neighbour_x
     * @param neighbour_y
     * @return
     */
    static bool rule_1_inbounds(Mat& group_pixels, int neighbour_x, int neighbour_y);
    /**
     * @brief rule_2_unoccupied Check if pixel does not already belong to a segment
     * @param group_pixels
     * @param neighbour_x
     * @param neighbour_y
     * @return
     */
    static bool rule_2_unoccupied(Mat& group_pixels, int neighbour_x, int neighbour_y);
    /**
     * @brief rule_3_ndiff_inrange Check difference in neighbouring pixel values within threshold
     * @param depth_pixels
     * @param depth_param
     * @param x
     * @param y
     * @param neighbour_x
     * @param neighbour_y
     * @return
     */
    static bool rule_3_ndiff_inrange(Mat& depth_pixels, double depth_param, int x, int y, int neighbour_x, int neighbour_y);
    /**
     * @brief rule_4_depth_inrange Check if depth below max absolute difference threshold if new pixel added
     * @param depth_pixels
     * @param this_segment
     * @param max_depth
     * @param neighbour_x
     * @param neighbour_y
     * @return
     */
    static bool rule_4_depth_inrange(Mat& depth_pixels, Segment &this_segment, double max_depth, int neighbour_x, int neighbour_y);
    /**
     * @brief rule_5_width_inrange Check if width max threshold if new pixel added
     * @param this_segment
     * @param max_width
     * @param neighbour_x
     * @return
     */
    static bool rule_5_width_inrange(Segment& this_segment, double max_width, int neighbour_x);
    /**
     * @brief rule_6_height_inrange Check if height max threshold if new pixel added
     * @param this_segment
     * @param max_height
     * @param neighbour_y
     * @return
     */
    static bool rule_6_height_inrange(Segment& this_segment, double max_height, int neighbour_y);
    /**
     * @brief find_range Find length of interval formed by max, min, current
     * @param max
     * @param min
     * @param current
     * @return
     */
    static int find_range(int max, int min, int current);
    /**
     * @brief rational Compute a value using coefficients taken from rational function
     * @param series
     * @param variable
     * @return
     */
    static double rational(vector<double>& series, double variable);
    /**
     * @brief exponential Compute a value using coefficients taken from exponential function
     * @param series
     * @param variable
     * @return
     */
    static double exponential(vector<double>& series, double variable);

    /**
     * @brief Scale_value Scale value between scale_start scale_end relative to min_depth max_depth
     * @param to_be_scaled
     * @param compute_max_ndiff
     * @param min_depth
     * @param scale_start
     * @param scale_end
     * @return
     */
    static double Scale_value(double to_be_scaled, unsigned short compute_max_ndiff, unsigned short min_depth, double scale_start, double scale_end);
};

/**
 * @brief The Print_results class
 * Prints results
 */
class Print_results
{
public:
    /**
     * @brief print_segments_png Prints the results of segmentation into a PNG
     * @param input
     * @param file_name_path
     */
    static void print_segments_png(Processed_depth_map& input, string file_name_path);
    /**
     * @brief print_segments_csv Prints the results of segmentation into a CSV
     * @param input
     * @param file_name_path
     */
    static void print_segments_csv(Processed_depth_map& input, string file_name_path);
    /**
     * @brief print_features_csv Prints computed features as CSV
     * @param input
     * @param file_name_path
     */
    static void print_features_csv(Processed_depth_map& input, string file_name_path);
    /**
     * @brief print_viewable_png Prints the input depth map in a human viewable PNG
     * @param input
     * @param file_name_path
     */
    static void print_viewable_png(Processed_depth_map& input, string file_name_path);
protected:
    /**
     * @brief upper_bytes Constant used to choose colours
     */
    static const unsigned int upper_bytes = 0b11111111000000000000000000000000;
    /**
     * @brief lower_bytes Constant used to choose colours
     */
    static const unsigned int lower_bytes = 0b00000000111111111111111111111111;
};

#endif // QUICKFIND_H
