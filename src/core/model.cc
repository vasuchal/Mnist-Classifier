#include <core/model.h>
#include <core/image_processor.h>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>

namespace naivebayes {
    
    // initializes vector to size and sets elements to values of 0
    Model::Model(int image_size, int number_of_digits) 
    : image_size_(image_size), number_of_digits_(number_of_digits) {
        std::pair<int, int> shades_counter(0, 0);
        std::vector<std::pair<int, int>> pixel_cols(image_size_, shades_counter);
        std::vector<std::vector<std::pair<int, int>>> pixel_rows(image_size_, pixel_cols);
        std::vector<std::vector<std::vector<std::pair<int, int>>>> digit_classes(number_of_digits_, pixel_rows);
        image_processing_data_ = digit_classes;

        std::pair<float, float> probability_of_shades(0, 0);
        std::vector<std::pair<float, float>> cols(image_size_, probability_of_shades);
        std::vector<std::vector<std::pair<float, float>>> rows(image_size_, cols);
        std::vector<std::vector<std::vector<std::pair<float, float>>>> classes(number_of_digits_, rows);
        probability_pixel_data_ = classes;
    }
    
     std::istream &operator>>(std::istream &input, Model &model) {
        std::vector<ImageProcessor> images; 
        ImageProcessor image_processor = ImageProcessor(model.image_size_);
        while (input >> image_processor) {
            images.push_back(image_processor);
        }
        
        model.total_image_count_ = images.size();
        for (ImageProcessor image : images) {
            int class_number = image.digit();
            for (size_t row = 0; row < image.pixels().size(); row++) {
                for (size_t col = 0; col < image.pixels().size(); col++) {
                    if (image.pixels()[row][col] == 1) {
                        model.image_processing_data_[class_number][row][col].first++;
                    } else {
                        model.image_processing_data_[class_number][row][col].second++;
                    }
                }
            }
        }

        return input;
    }

    void Model::Train() {
        int digit_images_count = 0; 
        for (size_t digit_class = 0; digit_class < image_processing_data_.size(); digit_class++) {
            for (size_t row = 0; row < image_processing_data_[digit_class].size(); row++) {
                for (size_t col = 0; col < image_processing_data_[digit_class][row].size(); col++) {
                    float shaded_count = image_processing_data_[digit_class][row][col].first;
                    float unshaded_count = image_processing_data_[digit_class][row][col].second;
                    digit_images_count = shaded_count + unshaded_count;
                    float shaded_pixel_probability = (shaded_count + kLaplaceFactor) / 
                            (digit_images_count + 2 * kLaplaceFactor);
                    float unshaded_pixel_probability = 1 - shaded_pixel_probability;
                    probability_pixel_data_[digit_class][row][col].first = shaded_pixel_probability;
                    probability_pixel_data_[digit_class][row][col].second = unshaded_pixel_probability;
                }
            }
            float class_probability = (float) digit_images_count /total_image_count_; 
            class_probabilities_.push_back(class_probability); 
        }
    }

    void Model::Save(std::string file_path) const {
        std::ofstream my_file(file_path);
        if (my_file.is_open()) {
            for (size_t digit_class = 0; digit_class < probability_pixel_data_.size(); digit_class++) {
                for (size_t row = 0; row < probability_pixel_data_[digit_class].size(); row++) {
                    for (size_t col = 0; col < probability_pixel_data_[digit_class][row].size(); col++) {
                        float shaded_probability = probability_pixel_data_[digit_class][row][col].first;
                        // read only one shade's probabilities because other one's can be determined 
                        // by subtracting this value from 1
                        my_file << shaded_probability << std::endl;
                    }
                }
            }

            my_file << "Class Probabilities" << std::endl;
            for (size_t digit = 0; digit < class_probabilities_.size(); digit++) {
                my_file << class_probabilities_[digit] << std::endl;
            }
            my_file.close();
        }
        else std::cout << "Unable to open file";
    }

    void Model::Load(std::string file_path) {
        std::string line;
        std::ifstream my_file(file_path);
        int digit_class = 0;
        int row = 0;
        int col = 0;
        bool is_class_probabilities = false;
        while (getline( my_file, line)) {
            
            if (is_class_probabilities) {
                class_probabilities_[digit_class] = std::stof(line);
                digit_class++;
                continue;
            }

            if (line.find_first_of("Class Probabilities") == 0) {
                is_class_probabilities = true;
                std::vector<float> init_vec(10, 0);
                digit_class = 0;
                class_probabilities_ = init_vec;
                continue;
            }
            
            if (col == image_size_) {
                col = 0;
                row++;
            }
            if (row == image_size_) {
                row = 0;
                digit_class++;
            }
            probability_pixel_data_[digit_class][row][col].first = std::stof(line);
            probability_pixel_data_[digit_class][row][col].second = 
                    1 - probability_pixel_data_[digit_class][row][col].first;
            col++;
        }
        my_file.close();
    }

    std::pair<int, int> Model::GetShadeCountsByIndex(int class_index, int row, int col) const {
        return image_processing_data_[class_index][row][col];
    }

    std::pair<float, float> Model::GetProbabilitiesByIndex(int class_index, int row, int col) const {
        return probability_pixel_data_[class_index][row][col];
    }
    
    float Model::GetClassProbabilityByIndex(int index) const{
        return class_probabilities_[index];
    }

}  // namespace naivebayes
