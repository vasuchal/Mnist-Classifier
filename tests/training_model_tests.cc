#include <catch2/catch.hpp>
#include <core/model.h>
#include <fstream>

TEST_CASE("Testing overloading operator and probabilities") {
    std::ifstream my_file( "/Users/vasudhachalasani/CLionProjects/Cinder/my-projects/naive-bayes-vasuchal/data/testing/testtrainingdata.txt" );
    naivebayes::Model model(3, 2);
    my_file >> model;
    my_file.close();
    model.Train();
    model.Save("/Users/vasudhachalasani/CLionProjects/Cinder/my-projects/naive-bayes-vasuchal/data/testing/saved_model_data.txt");
    SECTION("Testing overloading operator for shaded pixels count for digit 0") {
        std::vector<int> row_one = { 1, 1, 1 };
        std::vector<int> row_two = { 1, 0, 1 };
        std::vector<int> row_three = { 1, 1, 1 };
        std::vector<std::vector<int>> shaded_counts = {row_one, row_two, row_three};
        for (size_t row = 0; row < shaded_counts.size(); row++) {
            for (size_t col = 0; col < shaded_counts.size(); col++) {
                REQUIRE(model.GetShadeCountsByIndex(0,row,col).first == shaded_counts[row][col]);
            }
        }
    }

    SECTION("Testing overloading operator for unshaded pixels count for digit 0") {
        std::vector<int> row_one = { 0, 0, 0 };
        std::vector<int> row_two = { 0, 1, 0 };
        std::vector<int> row_three = { 0, 0, 0 };
        std::vector<std::vector<int>> unshaded_counts = {row_one, row_two, row_three};
        for (size_t row = 0; row < unshaded_counts.size(); row++) {
            for (size_t col = 0; col < unshaded_counts.size(); col++) {
                REQUIRE(model.GetShadeCountsByIndex(0,row,col).second == unshaded_counts[row][col]);
            }
        }
    }

    SECTION("Testing overloading operator for shaded pixels count for digit 1") {

        std::vector<int> row_one = { 1, 2, 0 };
        std::vector<int> row_two = { 0, 2, 0 };
        std::vector<int> row_three = { 1, 2, 1 };
        std::vector<std::vector<int>> shaded_counts = {row_one, row_two, row_three};
        for (size_t row = 0; row < shaded_counts.size(); row++) {
            for (size_t col = 0; col < shaded_counts.size(); col++) {
                REQUIRE(model.GetShadeCountsByIndex(1,row,col).first == shaded_counts[row][col]);
            }
        }
    }

    SECTION("Testing overloading operator for unshaded pixels count for digit 1") {
        std::vector<int> row_one = { 1, 0, 2 };
        std::vector<int> row_two = { 2, 0, 2 };
        std::vector<int> row_three = { 1, 0, 1 };
        std::vector<std::vector<int>> unshaded_counts = {row_one, row_two, row_three};
        for (size_t row = 0; row < unshaded_counts.size(); row++) {
            for (size_t col = 0; col < unshaded_counts.size(); col++) {
                REQUIRE(model.GetShadeCountsByIndex(1,row,col).second == unshaded_counts[row][col]);
            }
        }
    }

    SECTION("Testing train method shaded pixel probabilities for digit 0") {
        std::vector<float> row_one = { 0.66667f, 0.66667f, 0.66667f };
        std::vector<float> row_two = { 0.66667f, 0.33333f, 0.66667f };
        std::vector<float> row_three = { 0.66667f, 0.66667f, 0.66667f };
        std::vector<std::vector<float>> shaded_probabilities = {row_one, row_two, row_three};
        for (size_t row = 0; row < shaded_probabilities.size(); row++) {
            for (size_t col = 0; col < shaded_probabilities.size(); col++) {
                REQUIRE(model.GetProbabilitiesByIndex(0,row,col).first == Approx(shaded_probabilities[row][col]));
            }
        }
        
    }

    SECTION("Testing train method unshaded pixel probabilities for digit 0") {
        std::vector<float> row_one = { 0.33333f, 0.33333f, 0.33333f};
        std::vector<float> row_two = { 0.33333f, 0.66667f, 0.33333f};
        std::vector<float> row_three = { 0.33333f, 0.33333f, 0.33333f };
        std::vector<std::vector<float>> unshaded_probabilities = {row_one, row_two, row_three};
        for (size_t row = 0; row < unshaded_probabilities.size(); row++) {
            for (size_t col = 0; col < unshaded_probabilities.size(); col++) {
                REQUIRE(model.GetProbabilitiesByIndex(0,row,col).second == Approx(unshaded_probabilities[row][col]));
            }
        }
    }

    SECTION("Testing train method for shaded pixels probabilities for digit 1") {
        std::vector<float> row_one = { 0.5f, 0.75f, 0.25f };
        std::vector<float> row_two = { 0.25f, 0.75f, 0.25f };
        std::vector<float> row_three = { 0.5f, 0.75f, 0.5f };
        std::vector<std::vector<float>> shaded_probabilities = {row_one, row_two, row_three};
        for (size_t row = 0; row < shaded_probabilities.size(); row++) {
            for (size_t col = 0; col < shaded_probabilities.size(); col++) {
                REQUIRE(model.GetProbabilitiesByIndex(1,row,col).first == Approx(shaded_probabilities[row][col]));
            }
        }
    }

    SECTION("Testing train method for unshaded pixels probabilities for digit 1") {
        std::vector<float> row_one = { 0.5f, 0.25f, 0.75f};
        std::vector<float> row_two = { 0.75f, 0.25f, 0.75f};
        std::vector<float> row_three = { 0.5f, 0.25f, 0.5f };
        std::vector<std::vector<float>> unshaded_probabilities = {row_one, row_two, row_three};
        for (size_t row = 0; row < unshaded_probabilities.size(); row++) {
            for (size_t col = 0; col < unshaded_probabilities.size(); col++) {
                REQUIRE(model.GetProbabilitiesByIndex(1,row,col).second == Approx(unshaded_probabilities[row][col]));
            }
        }
    }
    
    SECTION("Testing train method for class probabilities") {
        std::vector<float> class_probabilities = { 0.33333f, 0.66667f};
        for (size_t index = 0; index < class_probabilities.size(); index++) {
            REQUIRE(model.GetClassProbabilityByIndex(index) == Approx(class_probabilities[index]));
        }
    }
}

TEST_CASE("Testing save and load functions") {
    naivebayes::Model loaded_model(3, 2);
    loaded_model.Load("/Users/vasudhachalasani/CLionProjects/Cinder/my-projects/naive-bayes-vasuchal/data/testing/saved_model_data.txt");
    
    SECTION("Testing save and load method for shaded pixel probabilities of digit 0") {
        std::vector<float> row_one = { 0.66667f, 0.66667f, 0.66667f };
        std::vector<float> row_two = { 0.66667f, 0.33333f, 0.66667f };
        std::vector<float> row_three = { 0.66667f, 0.66667f, 0.66667f };
        std::vector<std::vector<float>> shaded_probabilities = {row_one, row_two, row_three};
        for (size_t row = 0; row < shaded_probabilities.size(); row++) {
            for (size_t col = 0; col < shaded_probabilities.size(); col++) {
                REQUIRE(loaded_model.GetProbabilitiesByIndex(0,row,col).first == Approx(shaded_probabilities[row][col]));
            }
        }
    }

    SECTION("Testing save and load method for unshaded pixel probabilities of digit 0") {
        std::vector<float> row_one = { 0.33333f, 0.33333f, 0.33333f};
        std::vector<float> row_two = { 0.33333f, 0.66667f, 0.33333f};
        std::vector<float> row_three = { 0.33333f, 0.33333f, 0.33333f };
        std::vector<std::vector<float>> unshaded_probabilities = {row_one, row_two, row_three};
        for (size_t row = 0; row < unshaded_probabilities.size(); row++) {
            for (size_t col = 0; col < unshaded_probabilities.size(); col++) {
                REQUIRE(loaded_model.GetProbabilitiesByIndex(0,row,col).second == Approx(unshaded_probabilities[row][col]));
            }
        }
    }

    SECTION("Testing save and load method for shaded pixel probabilities of digit 1") {
        std::vector<float> row_one = { 0.5f, 0.75f, 0.25f };
        std::vector<float> row_two = { 0.25f, 0.75f, 0.25f };
        std::vector<float> row_three = { 0.5f, 0.75f, 0.5f };
        std::vector<std::vector<float>> shaded_probabilities = {row_one, row_two, row_three};
        for (size_t row = 0; row < shaded_probabilities.size(); row++) {
            for (size_t col = 0; col < shaded_probabilities.size(); col++) {
                REQUIRE(loaded_model.GetProbabilitiesByIndex(1,row,col).first == Approx(shaded_probabilities[row][col]));
            }
        }
    }

    SECTION("Testing save and load method for unshaded pixel probabilities of digit 1") {
        std::vector<float> row_one = { 0.5f, 0.25f, 0.75f};
        std::vector<float> row_two = { 0.75f, 0.25f, 0.75f};
        std::vector<float> row_three = { 0.5f, 0.25f, 0.5f };
        std::vector<std::vector<float>> unshaded_probabilities = {row_one, row_two, row_three};
        for (size_t row = 0; row < unshaded_probabilities.size(); row++) {
            for (size_t col = 0; col < unshaded_probabilities.size(); col++) {
                REQUIRE(loaded_model.GetProbabilitiesByIndex(1,row,col).second == Approx(unshaded_probabilities[row][col]));
            }
        }
    }
    
}
