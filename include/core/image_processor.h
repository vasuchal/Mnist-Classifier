#pragma once

#include <vector>

namespace naivebayes {
    using std::vector;
    
class ImageProcessor {
  public:
   /**
    * Constructor that takes image size
    */
   ImageProcessor(int image_size);
   
   /**
    * Constructor used for Cinder taking in a 2d vector
    */   
   ImageProcessor(const vector<vector<int>>& pixels);

   /**
    * Overloading operator to read images into format of 1s and 0s for shaded and unshaded
    */
   friend std::istream &operator>>(std::istream &input, ImageProcessor &imageProcessor);
   
   int digit() const;
   
   const vector<vector<int>>& pixels() const;
        
  private:
        vector<vector<int>> pixels_;
        int digit_;
        size_t image_size_;
    };


    
}

