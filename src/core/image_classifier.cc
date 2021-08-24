#include <core/image_classifier.h>
#include <fstream>
#include <core/image_processor.h>
#include <cmath>
#include <iostream>

namespace naivebayes {
    void ImageClassifier::Validate(std::string file_path, const Model &model) {
        std::ifstream input(file_path);
        std::vector<ImageProcessor> images;
        ImageProcessor image_processor = ImageProcessor(image_size_);
        
        while (input >> image_processor) {
            images.push_back(image_processor);
        }

        
        size_t total_images = images.size();
        int correct_prediction_count = 0;
        
        for (ImageProcessor image : images) {
            int prediction_value = Classify(image, model);
            int actual_value = image.digit();
            if (actual_value == prediction_value) {
                correct_prediction_count++;
            }
        }
        accuracy_ = (float) correct_prediction_count /  total_images;
        
        std::cout << accuracy_;
    }
    
    ImageClassifier::ImageClassifier(int image_size, int number_of_digits) 
    : image_size_(image_size), number_of_digits(number_of_digits) {
        std::vector<float> digit_probabilities(number_of_digits, 0);
        likelihood_scores_ = digit_probabilities;
    }
    
    int ImageClassifier::Classify(const ImageProcessor &image, const Model &model) {
        for (size_t digit = 0; digit < number_of_digits; digit++) {
            float pixel_probabilities = 0;
            for (size_t row = 0; row < image.pixels().size(); row++) {
                for (size_t col = 0; col < image.pixels().size(); col++) {
                    
                    float feature_probability = 0;
                    if (image.pixels()[row][col] == 1) {
                        feature_probability = (float) std::log(model.GetProbabilitiesByIndex(digit, row, col).first);
                    } else {
                        feature_probability = (float) std::log(model.GetProbabilitiesByIndex(digit, row, col).second);
                    }
                    
                    pixel_probabilities += feature_probability;
                }
            }
            
            float digit_likelihood_score = std::log(model.GetClassProbabilityByIndex(digit)) + pixel_probabilities;
              likelihood_scores_[digit] = digit_likelihood_score;

        }

        int prediction = std::distance(likelihood_scores_.begin(),std::max_element(likelihood_scores_.begin(),
                                                        likelihood_scores_.end()));
        return prediction;
    }

    float ImageClassifier::accuracy() const {
        return accuracy_;
    }

    float ImageClassifier::GetScoreByIndex(int index) const {
        return likelihood_scores_[index];
    }
}

