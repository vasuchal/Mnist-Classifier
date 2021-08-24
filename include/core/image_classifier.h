#pragma once

#include "model.h"
#include "image_processor.h"

namespace naivebayes {

class ImageClassifier {
  public:
   /**
    * Constructor taking image size and number of digits
    */
    ImageClassifier(int image_size, int number_of_digits);

   /**
    * Method returning digit image is classified as and takes in trained model and image as params
    */
    int Classify(const ImageProcessor& image, const Model& model);

   /**
    * Used to calculate accuracy on a file of images and takes in a trained model as a param
    */    
    void Validate(std::string file_path, const Model& model);
    
   /**
     * Used to get likelihood score by index
     */
    float GetScoreByIndex(int index) const;
    
    float accuracy() const;
    
  private:
    float accuracy_;
    vector<float> likelihood_scores_;
    size_t image_size_;
    size_t number_of_digits;
       
    };
}
