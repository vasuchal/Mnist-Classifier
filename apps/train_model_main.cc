#include <iostream>
#include <array>
#include <fstream>
#include <core/image_classifier.h>


int main() {
    std::ifstream my_file( "../data/trainingimagesandlabels.txt");
    naivebayes::Model model(28, 10);
    my_file >> model;
    my_file.close();
    model.Train();
    model.Save("/Users/vasudhachalasani/CLionProjects/Cinder/my-projects/naive-bayes-vasuchal/"
               "data/model_data.txt");
    model.Load("/Users/vasudhachalasani/CLionProjects/Cinder/my-projects/naive-bayes-vasuchal/data/model_data.txt");

    naivebayes::ImageClassifier imageClassifier(28, 10);
    imageClassifier.Validate("../data/testimagesandlabels.txt", model);

    return 0;
}
