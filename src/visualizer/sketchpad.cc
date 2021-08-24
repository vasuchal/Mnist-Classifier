#include <visualizer/sketchpad.h>

namespace naivebayes {

namespace visualizer {

using glm::vec2;

Sketchpad::Sketchpad(const vec2& top_left_corner, size_t num_pixels_per_side,
                     double sketchpad_size, double brush_radius)
    : top_left_corner_(top_left_corner),
      num_pixels_per_side_(num_pixels_per_side),
      pixel_side_length_(sketchpad_size / num_pixels_per_side),
      brush_radius_(brush_radius) {
    std::vector<int> level_two_vector(num_pixels_per_side_, 0);
    std::vector<std::vector<int>> level_one_vector(num_pixels_per_side_, level_two_vector);
    pixels_ = level_one_vector;
}

void Sketchpad::Draw() const {
  for (size_t row = 0; row < num_pixels_per_side_; ++row) {
    for (size_t col = 0; col < num_pixels_per_side_; ++col) {
      // Currently, this will draw a quarter circle centered at the top-left
      // corner with a radius of 20

      // if the pixel at (row, col) is currently shaded
      if (pixels_[row][col] == 1) {
        ci::gl::color(ci::Color("black"));
      } else {
        ci::gl::color(ci::Color("white"));
      }
      
      ci::Color color = pixels_[row][col] == 1 ? ci::Color("black") : ci::Color("white");
      ci::gl::color(color);
      
      vec2 pixel_top_left = top_left_corner_ + vec2(col * pixel_side_length_,
                                                    row * pixel_side_length_);

      vec2 pixel_bottom_right =
                pixel_top_left + vec2(pixel_side_length_, pixel_side_length_);
      ci::Rectf pixel_bounding_box(pixel_top_left, pixel_bottom_right);

      ci::gl::drawSolidRect(pixel_bounding_box); //white square with colors

      ci::gl::color(ci::Color("black")); //draws lines after choosing color
      ci::gl::drawStrokedRect(pixel_bounding_box); 
    }
  }
}

void Sketchpad::HandleBrush(const vec2& brush_screen_coords) {
  vec2 brush_sketchpad_coords =
      (brush_screen_coords - top_left_corner_) / (float)pixel_side_length_;

  for (size_t row = 0; row < num_pixels_per_side_; ++row) {
    for (size_t col = 0; col < num_pixels_per_side_; ++col) {
      vec2 pixel_center = {col + 0.5, row + 0.5};

      if (glm::distance(brush_sketchpad_coords, pixel_center) <=
          brush_radius_) {
          pixels_[row][col] = 1;
      }
    }
  }
}

void Sketchpad::Clear() {
        for (size_t row = 0; row < num_pixels_per_side_; ++row) {
            for (size_t col = 0; col < num_pixels_per_side_; ++col) {
                pixels_[row][col] = 0;
            }
        }
}

    const vector<vector<int>> &Sketchpad::pixels() const {
        return pixels_;
    }

}  // namespace visualizer

}  // namespace naivebayes
