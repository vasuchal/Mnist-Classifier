#include <catch2/catch.hpp>
#include <core/image_processor.h>
#include <core/image_classifier.h>

TEST_CASE("Testing digits classification probabilities and accuracies") {
    naivebayes::Model model(3, 2);
    model.Load("/Users/vasudhachalasani/CLionProjects/Cinder/my-projects/naive-bayes-vasuchal/data/testing/saved_model_data.txt");

    SECTION("Testing classification method for digit 0") {
        std::vector<int> row_one = { 1, 1, 1 };
        std::vector<int> row_two = { 1, 0, 1 };
        std::vector<int> row_three = { 1, 1, 1 };
        std::vector<std::vector<int>> image = {row_one, row_two, row_three};
        naivebayes::ImageProcessor imageProcessor(image);

        naivebayes::ImageClassifier imageClassifier(3, 2);
        REQUIRE(imageClassifier.Classify(imageProcessor, model) == 0);
    }
    
    SECTION("Testing classification method for digit 1") {
        std::vector<int> row_one = { 1, 1, 0 };
        std::vector<int> row_two = { 0, 1, 0 };
        std::vector<int> row_three = { 1, 1, 1 };
        std::vector<std::vector<int>> image = {row_one, row_two, row_three};
        naivebayes::ImageProcessor imageProcessor(image);

        naivebayes::ImageClassifier imageClassifier(3, 2);
        REQUIRE(imageClassifier.Classify(imageProcessor, model) == 1);
    }

    SECTION("Testing classification probabilities for digit 0") {
        std::vector<int> row_one = { 1, 1, 1 };
        std::vector<int> row_two = { 1, 0, 1 };
        std::vector<int> row_three = { 1, 1, 1 };
        std::vector<std::vector<int>> image = {row_one, row_two, row_three};
        naivebayes::ImageProcessor imageProcessor(image);

        naivebayes::ImageClassifier imageClassifier(3, 2);
        imageClassifier.Classify(imageProcessor, model);
        REQUIRE(imageClassifier.GetScoreByIndex(0) == Approx(-4.7478f));
        REQUIRE(imageClassifier.GetScoreByIndex(1) == Approx(-8.60545f));
    }

    SECTION("Testing classification probabilities for digit 1") {
        std::vector<int> row_one = { 1, 1, 0 };
        std::vector<int> row_two = { 0, 1, 0 };
        std::vector<int> row_three = { 1, 1, 1 };
        std::vector<std::vector<int>> image = {row_one, row_two, row_three};
        naivebayes::ImageProcessor imageProcessor(image);

        naivebayes::ImageClassifier imageClassifier(3, 2);
        imageClassifier.Classify(imageProcessor, model);
        REQUIRE(imageClassifier.GetScoreByIndex(0) == Approx(-7.52039f));
        REQUIRE(imageClassifier.GetScoreByIndex(1) == Approx(-4.211f));
    }

    SECTION("Testing accuracy for digits 0-9") {
        naivebayes::Model n_model(28, 10);
        n_model.Load("/Users/vasudhachalasani/CLionProjects/Cinder/my-projects/naive-bayes-vasuchal/data/model_data.txt");

        naivebayes::ImageClassifier imageClassifier(28, 10);
        imageClassifier.Validate("/Users/vasudhachalasani/CLionProjects/Cinder/my-projects/naive-bayes-vasuchal//data/testimagesandlabels.txt", n_model);
        REQUIRE(imageClassifier.accuracy() == Approx(0.771f));
    }
    
}



