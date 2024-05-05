#include "quickfind.h"

Segment::Segment(Mat& depth_map, Mat& group_pixels, int group_id, int block_across, int block_down, tuple<int, int, unsigned short> start, double scale_start, double scale_end)
{
    this->depth_map = depth_map;
    this->group_pixels = group_pixels;
    this->group_id = group_id;
    this->block_across = block_across;
    this->block_down = block_down;
    this->contains_valid_pixels = false;
    this->process = false;
    this->scale_start = scale_start;
    this->scale_end = scale_end;
    this->pixel_count = 0;
    this->add(start);
}

void Segment::add(tuple<int, int, unsigned short> added_pixel)
{
    // Do not add any more pixels if finalised for processing
    if (this->process)
    {
        return;
    }

    if (this->pixel_count > 0)
    {
        if (get<0>(this->max_y) < get<0>(added_pixel)) { this->max_y = added_pixel; }
        if (get<0>(this->min_y) > get<0>(added_pixel)) { this->min_y = added_pixel; }
        if (get<1>(this->max_x) < get<1>(added_pixel)) { this->max_x = added_pixel; }
        if (get<1>(this->min_x) > get<1>(added_pixel)) { this->min_x = added_pixel; }
        if (get<2>(this->max_depth) < get<2>(added_pixel)) { this->max_depth = added_pixel; }
        if (get<2>(this->min_depth) > get<2>(added_pixel)) { this->min_depth = added_pixel; }
    }
    else
    {
        this->max_x = added_pixel;
        this->min_x = added_pixel;
        this->max_y = added_pixel;
        this->min_y = added_pixel;
        this->max_depth = added_pixel;
        this->min_depth = added_pixel;
    }
    // Increment pixel counter
    ++(this->pixel_count);
    if (get<2>(added_pixel) > 0)
    {
        this->contains_valid_pixels = true;
    }
}

void Segment::produce_segment_map()
{
    int width = get<1>(this->max_x) - get<1>(this->min_x) + 1;
    int height = get<0>(this->max_y) - get<0>(this->min_y) + 1;
    int startx = get<1>(this->min_x);
    int starty = get<0>(this->min_y);

    // Make copy of pixels belonging to segment, used for feature computation
    this->segment_map = Mat::zeros(height, width, CV_16UC1);

    // Get the regions containing segment
    Rect segment_section = Rect(startx, starty, width, height);
    Mat group_sub = this->group_pixels(segment_section);
    Mat depth_sub = this->depth_map(segment_section);
    // Copy the pixels of segment
    Mat group_mask;
    inRange(group_sub, this->group_id, this->group_id, group_mask);
    depth_sub.copyTo(this->segment_map, group_mask);
}

bool Segment::Should_finalise()
{
    // Discard segments which contain only nonsense/zero data
    bool not_zero = this->contains_valid_pixels;
    // Discard segments below size threshold
    int width = (get<1>(this->max_x) - get<1>(this->min_x) + 1);
    int height = (get<0>(this->max_y) - get<0>(this->min_y) + 1);
    bool width_ok = width > this->block_across;
    bool height_ok = height > this->block_across;

    // If this flag is true then this is valid segment for processing
    this->process = not_zero && width_ok && height_ok;
    return (this->process);
}

void Segment::finalise()
{
    // Produce a mini depth map of just this segment
    this->produce_segment_map();

    // Compute features
    int max_y = get<0>(this->max_y);
    int min_y = get<0>(this->min_y);
    int max_x = get<1>(this->max_x);
    int min_x = get<1>(this->min_x);
    unsigned short max_depth = get<2>(this->max_depth);
    unsigned short min_depth = get<2>(this->min_depth);
    Algorithms::Compute_dimensions(this->features, this->pixel_count, min_x, max_x, min_y, max_y, min_depth, max_depth);
    Algorithms::Compute_blocks(this->features, this->segment_map, this->block_across, this->block_down,  min_x, max_x, min_y, max_y, min_depth, max_depth, this->scale_start, this->scale_end);
}

Processed_depth_map::Processed_depth_map(Mat& depth_map, int block_across, int block_down, vector<double>& width, vector<double>& height, vector<double>& depth, vector<double>& ndiff, double scale_start, double scale_end)
{
    this->depth_map = depth_map;
    this->block_across = block_across;
    this->block_down = block_down;
    this->ndiff = ndiff;
    this->depth = depth;
    this->width = width;
    this->height = height;
    this->scale_start = scale_start;
    this->scale_end = scale_end;

    // Created as part of the segmentation
    this->group_pixels = Mat::zeros(depth_map.size(), CV_32S);
    this->group_id = 1;
}

void Processed_depth_map::add_segment(Segment& new_segment)
{
    if (new_segment.Should_finalise())
    {
        new_segment.finalise();
        // Save features
        this->segments.push_back(new_segment);
    }
    ++(this->group_id);
}

Segment Algorithms::Connect_component_mod(Mat& depth_pixels, Mat& group_pixels, tuple<int, int, unsigned short> start, int block_across, int block_down, int group_id, vector<double> width, vector<double> height, vector<double> depth, vector<double> ndiff, double scale_start, double scale_end)
{
    // Initialise a new segment and mark starting pixel
    Segment this_segment = Segment(depth_pixels, group_pixels, group_id, block_across, block_down, start, scale_start, scale_end);
    group_pixels.at<int>(get<0>(start), get<1>(start)) = group_id;

    // The dimensions are capped by the starting depth pixel, precompute these values here
    unsigned short first_depth = get<2>(start);
    double max_width = Algorithms::compute_max_width(width, first_depth);
    double max_height = Algorithms::compute_max_height(height, first_depth);
    double max_depth = Algorithms::compute_max_depth(depth, first_depth);
    double max_ndiff = Algorithms::compute_max_ndiff(ndiff, first_depth);

    // Store the starting pixel for processing
    deque<tuple<int, int, unsigned short>> pixel_queue;
    pixel_queue.push_back(start);

    while (!pixel_queue.empty())
    {
        // Get first pixel from queue
        tuple<int, int, unsigned short> current = pixel_queue.front();
        pixel_queue.pop_front();

        int x = get<1>(current);
        int y = get<0>(current);

        // Check the if neighbouring pixels belong to this segment
        size_t indices = 8;
        int xs[] = { x - 1, x, x + 1, x + 1, x + 1, x, x - 1, x - 1 };
        int ys[] = { y - 1, y - 1, y - 1, y, y + 1, y + 1, y + 1, y };

        for (size_t i = 0; i < indices; ++i)
        {
            int neighbour_x = xs[i];
            int neighbour_y = ys[i];

            if (
                // Rule 1 must be executed first to perform range checking, otherwise algorithm will crash
                   Algorithms::rule_1_inbounds(group_pixels, neighbour_x, neighbour_y)
                && Algorithms::rule_2_unoccupied(group_pixels, neighbour_x, neighbour_y)
                && Algorithms::rule_3_ndiff_inrange(depth_pixels, max_ndiff, x, y, neighbour_x, neighbour_y)
                && Algorithms::rule_4_depth_inrange(depth_pixels, this_segment, max_depth, neighbour_x, neighbour_y)
                && Algorithms::rule_5_width_inrange(this_segment, max_width, neighbour_x)
                && Algorithms::rule_6_height_inrange(this_segment, max_height, neighbour_y)
               )
            {
                unsigned short neighbour_depth = depth_pixels.at<unsigned short>(neighbour_y, neighbour_x);
                tuple<int, int, unsigned short> neighbour { neighbour_y, neighbour_x, neighbour_depth };
                pixel_queue.push_back(neighbour);
                this_segment.add(neighbour);
                group_pixels.at<int>(neighbour_y, neighbour_x) = group_id;
            }
        }
    }
    return this_segment;
}

Processed_depth_map Algorithms::Segmentation(Mat& depth_pixels, int block_across, int block_down, vector<double>& width, vector<double>& height, vector<double>& depth, vector<double>& ndiff, double scale_start, double scale_end)
{
    Processed_depth_map my_processed_dm = Processed_depth_map(depth_pixels, block_across, block_down, width, height, depth, ndiff, scale_start, scale_end);

    for (int j = 0; j < my_processed_dm.depth_map.rows; ++j)
    {
        for (int i = 0; i < my_processed_dm.depth_map.cols; ++i)
        {
            if (my_processed_dm.group_pixels.at<int>(j, i) == 0)
            {
                tuple<int, int, unsigned short> starting_pixel { j, i, my_processed_dm.depth_map.at<unsigned short>(j, i) };
                Segment current_segment =
                Algorithms::Connect_component_mod(
                                                  my_processed_dm.depth_map,
                                                  my_processed_dm.group_pixels,
                                                  starting_pixel,
                                                  my_processed_dm.block_across,
                                                  my_processed_dm.block_down,
                                                  my_processed_dm.group_id,
                                                  my_processed_dm.width,
                                                  my_processed_dm.height,
                                                  my_processed_dm.depth,
                                                  my_processed_dm.ndiff,
                                                  my_processed_dm.scale_start,
                                                  my_processed_dm.scale_end
                                                 );
                // Segment added if valid
                my_processed_dm.add_segment(current_segment);
            }
        }
    }

    return my_processed_dm;
}

int Algorithms::find_range(int max, int min, int current)
{
    int interval = max - min + 1;
    if (current > max) { interval = current - min + 1; }
    if (current < min) { interval = max - current + 1; }
    return interval;
}

double Algorithms::rational(vector<double>& series, double variable)
{
    double sum = 0;
    if (variable == 0)
    {
        sum = numeric_limits<double>::max();
    }
    else
    {
        sum = series.at(0) + (series.at(1) / variable);
    }
    return sum;
}

double Algorithms::exponential(vector<double>& series, double variable)
{
    double sum = exp(series.at(0) + series.at(1) * variable);
    return sum;
}

double Algorithms::compute_max_width(vector<double>& width, unsigned short start_depth)
{
    double width_param = Algorithms::rational(width, static_cast<double>(start_depth));
    return (width_param);
}

double Algorithms::compute_max_height(vector<double>& height, unsigned short start_depth)
{
    double height_param = Algorithms::rational(height, static_cast<double>(start_depth));
    return (height_param);
}

double Algorithms::compute_max_depth(vector<double>& depth, unsigned short start_depth)
{
    double depth_param = Algorithms::rational(depth, static_cast<double>(start_depth));
    return (depth_param);
}

double Algorithms::compute_max_ndiff(vector<double>& ndiff, unsigned short start_depth)
{
    double ndiff_param = Algorithms::exponential(ndiff, static_cast<double>(start_depth));
    return (ndiff_param);
}

bool Algorithms::rule_1_inbounds(Mat& group_pixels, int neighbour_x, int neighbour_y)
{
    // Check if pixel within image boundary
    bool condition = (neighbour_x >= 0 && neighbour_x < group_pixels.cols && neighbour_y >= 0 && neighbour_y < group_pixels.rows);
    return condition;
}

bool Algorithms::rule_2_unoccupied(Mat& group_pixels, int neighbour_x, int neighbour_y)
{
    // Check if pixel occupied
    bool condition = (group_pixels.at<int>(neighbour_y, neighbour_x) == 0);
    return condition;
}

bool Algorithms::rule_3_ndiff_inrange(Mat& depth_pixels, double ndiff_threshold, int x, int y, int neighbour_x, int neighbour_y)
{
    // Check if difference between neighbouring pixel values below threshold
    unsigned short neighbour_value = depth_pixels.at<unsigned short>(neighbour_y, neighbour_x);
    unsigned short current_value = depth_pixels.at<unsigned short>(y, x);
    bool condition = abs(neighbour_value - current_value) <= ndiff_threshold;
    return condition;
}

bool Algorithms::rule_4_depth_inrange(Mat& depth_pixels, Segment &this_segment, double max_depth, int neighbour_x, int neighbour_y)
{
    // Check if depth below threshold if new pixel added
    unsigned short us_max_depth = get<2>(this_segment.max_depth);
    unsigned short us_min_depth = get<2>(this_segment.min_depth);
    unsigned short us_neighbour_depth = depth_pixels.at<unsigned short>(neighbour_y, neighbour_x);
    int depth_with_neighbour = Algorithms::find_range(us_max_depth, us_min_depth, us_neighbour_depth);
    bool condition = depth_with_neighbour <= max_depth;
    return condition;
}

bool Algorithms::rule_5_width_inrange(Segment& this_segment, double max_width, int neighbour_x)
{
    // Check if width below threshold if new pixel added
    int width_with_neighbour = Algorithms::find_range(get<1>(this_segment.max_x), get<1>(this_segment.min_x), neighbour_x);
    bool condition = width_with_neighbour <= max_width;
    return condition;
}

bool Algorithms::rule_6_height_inrange(Segment& this_segment, double max_height, int neighbour_y)
{
    // Check if height below threshold if new pixel added
    unsigned short us_max_y = get<0>(this_segment.max_y);
    unsigned short us_min_y = get<0>(this_segment.min_y);
    int height_with_neighbour = Algorithms::find_range(us_max_y, us_min_y, neighbour_y);
    bool condition = height_with_neighbour <= max_height;
    return condition;
}

double Algorithms::Scale_value(double to_be_scaled, unsigned short max_depth, unsigned short min_depth, double scale_start, double scale_end)
{
    double after_scale = scale_start;
    // In case the max and min are equal, avoid division by zero
    // Scaling and input values also must be checked
    if ((max_depth - min_depth > 0) && (scale_end - scale_start > 0) && (to_be_scaled > min_depth))
    {
        double scale_factor = double(scale_end - scale_start + 1);
        after_scale = scale_factor * double(to_be_scaled - min_depth) / double(max_depth - min_depth);
    }
    return after_scale;
}

void Algorithms::Compute_blocks(vector<double>& blocks, Mat segment_map, int block_across, int block_down, int min_x, int max_x, int min_y, int max_y, unsigned short min_depth, unsigned short max_depth, double scale_start, double scale_end)
{
    // Find the step increments for each block
    int width = max_x - min_x + 1;
    int height = max_y - min_y + 1;
    int step_width =  width / block_across;
    int step_height = height / block_down;
    for (int j = 0; j < height - step_height; j += step_height)
    {
        for (int i = 0; i < width - step_width; i += step_width)
        {
            // Compute mean block values then save and store them
            Rect subsection = Rect(i, j, step_width, step_height);
            Mat segment_sub = segment_map(subsection);
            Scalar temp_mean = cv::mean(segment_sub);
            double mean = temp_mean.val[0];
            double mean_scaled = Algorithms::Scale_value(mean, max_depth, min_depth, scale_start, scale_end);
            blocks.push_back(mean_scaled);
        }
    }
}

void Algorithms::Compute_dimensions(vector<double>& dimensions, unsigned long pixel_count, int min_x, int max_x, int min_y, int max_y, unsigned short min_depth, unsigned short max_depth)
{
    double size = static_cast<double>(pixel_count);
    double width = static_cast<double>(max_x - min_x + 1);
    double height = static_cast<double>(max_y - min_y + 1);
    double depth = static_cast<double>(max_depth - min_depth + 1);
    dimensions.push_back(size);
    dimensions.push_back(width);
    dimensions.push_back(height);
    dimensions.push_back(depth);
}

void Print_results::print_segments_png(Processed_depth_map& input, string file_name_path)
{
    Mat colours = Mat::zeros(input.group_pixels.size(), CV_8UC4);
    // Each pixel is assigned a colour based on number of segments
    unsigned int colour_comp = Print_results::lower_bytes / static_cast<unsigned int>(input.group_id + 1);
    for (int j = 0; j < colours.rows; ++j)
    {
        for (int i = 0; i < colours.cols; ++i)
        {
            // Get Colours
            int group_id = input.group_pixels.at<int>(j, i);
            unsigned int pixel_colour = static_cast<unsigned int>(group_id) * colour_comp;
            // The first 24 bits are colour channels the last 8 bits are always maxed out to ensure max alpha channel
            pixel_colour = pixel_colour | Print_results::upper_bytes;
            // Write colours
            colours.at<unsigned int>(j, i) = pixel_colour;
        }
    }
    imwrite(file_name_path, colours);
}

void Print_results::print_segments_csv(Processed_depth_map& input, string file_name_path)
{
    ofstream file;
    file.open(file_name_path);
    for (int j = 0; j < input.group_pixels.rows; ++j)
    {
        for (int i = 0; i < input.group_pixels.cols; ++i)
        {
            int group_id = input.group_pixels.at<int>(j, i);
            file << group_id;
            if (i < input.group_pixels.cols - 1)
            {
                file << ", ";
            }
        }
        file << "\n";
    }
    file.close();
}

void Print_results::print_features_csv(Processed_depth_map& input, string file_name_path)
{
    ofstream file;
    file.open(file_name_path);
    // Write header
    file << "Group ID, Size, Width, Height, Depth";
    for (size_t k = 0; k < size_t(input.block_across * input.block_down); ++k)
    {
        file << ", " << k;
    }
    file << "\n";
    // Write data
    for (size_t j = 0; j < input.segments.size(); ++j)
    {
        Segment current_seg = input.segments.at(j);
        // Only process valid segments
        if (current_seg.process)
        {
            file << current_seg.group_id << ", ";
            vector<double> current_feature = current_seg.features;
            for (size_t i = 0; i < current_feature.size(); ++i)
            {
                file << current_feature.at(i);
                if (i < current_feature.size() - 1)
                {
                    file << ", ";
                }
            }
            file << "\n";
        }
    }
    file.close();
}

void Print_results::print_viewable_png(Processed_depth_map& input, string file_name_path)
{
    // Find min and max value
    double min, max;
    minMaxLoc(input.depth_map, &min, &max);
    // Scale image values
    Mat viewable = Mat::zeros(input.depth_map.size(), CV_8UC1);
    // If input is blank image or all pixel values equal then print blank image
    // Otherwise scale image
    if (max > min)
    {
        viewable = numeric_limits<unsigned short>::max() * (input.depth_map - min) / (max - min);
    }
    imwrite(file_name_path, viewable);
}
