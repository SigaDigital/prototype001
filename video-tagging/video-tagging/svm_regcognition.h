#pragma once
#include "face-recognition.h"

using namespace dlib;
using namespace std;
#include <dlib/clustering.h>

typedef matrix<float, 0, 1> sample_type;
typedef radial_basis_kernel<sample_type> kernel_type;
typedef probabilistic_decision_function<kernel_type> probabilistic_funct_type;
typedef normalized_function<probabilistic_funct_type> pfunct_type;

class SvmRegcognition: public FaceRecognition
{
public:
	SvmRegcognition(char* path);
	string Recognize(cv::Mat& mat);
private:
	string path;
    string train_path;
	std::vector<string> name;
	std::vector<string> all_test;
	std::vector<pfunct_type> all_pairs;
};

