#include <string>
#include <vector>

namespace naivebayes {
    
class Model {
 public:
  /**
   * Constructor that takes in image width and height dimensions in variable image_size
   * and the number of digit classes
   */
  Model(int image_size, int number_of_digits);
  
  /**
   * Trains the model by calculating probabilities of each shaded and unshaded feature of every pixel
   * for every digit
   */  
  void Train();

  /**
   * Saves the model to a file of inputted file path
   */
  void Save(std::string file_path) const;
  
  /**
   * Loads training data from file passed in through its file path to the model 
   */  
  void Load(std::string file_path);

  /**
   * Reads data from stream input to model's image_processing_data variable 
   */
  friend std::istream &operator>>(std::istream &input, Model &model);
  
  /**
   * Returns pair of counts of shaded and unshaded with shaded counts being the first value 
   * and unshaded count being the second value in the pair
   */
  std::pair<int, int> GetShadeCountsByIndex(int class_index, int row, int col) const;

  /**
    * Returns pair of probabilities of shaded and unshaded features with shaded probabilities being the first value
    * and unshaded probabilities being the second value
    */
  std::pair<float, float> GetProbabilitiesByIndex(int class_index, int row, int col) const;

  /**
    * Returns class probability of selected index
    */
  float GetClassProbabilityByIndex(int index) const;

 private:
  std::vector<std::vector<std::vector<std::pair<int, int>>>> image_processing_data_;
  std::vector<std::vector<std::vector<std::pair<float, float>>>> probability_pixel_data_;
  std::vector<float> class_probabilities_;
  size_t image_size_;
  size_t number_of_digits_;
  size_t total_image_count_;
  const float kLaplaceFactor = 1;
  
};
}  // namespace naivebayes

