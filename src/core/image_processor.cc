#include <core/image_processor.h>
#include <istream>

namespace naivebayes {
    ImageProcessor::ImageProcessor(int image_size) : image_size_(image_size) {
        std::vector<int> level_two_vector(image_size_, 0);
        std::vector<std::vector<int>> level_one_vector(image_size_, level_two_vector);
        pixels_ = level_one_vector;
    }

    ImageProcessor::ImageProcessor(const vector<vector<int>>& pixels) {
        pixels_ = pixels;
    }
    
    std::istream &operator>>(std::istream &input, ImageProcessor &image_processor) {
    
    //input >> image_processor.digit_., new line, auto cast to int
    std::string line;
    getline(input, line);
    if (isdigit(line[0])) {
        image_processor.digit_ = (int) (line[0] - '0');
    }

    for (size_t row = 0; row < image_processor.pixels_.size(); row++) {
        getline(input, line);
        for (size_t col = 0; col < image_processor.pixels_.size(); col++) {
            if (line[col] == '+' || line[col] == '#') {
            image_processor.pixels_[row][col] = 1;
            } else {
                image_processor.pixels_[row][col] = 0;
            }
        }
    }

    return input;

}

    int ImageProcessor::digit() const{
        return digit_;
    }

    const vector<vector<int>> &ImageProcessor::pixels() const {
        return pixels_;
    }
    
}

