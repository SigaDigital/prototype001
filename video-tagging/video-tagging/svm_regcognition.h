#pragma once
#include "face-recognition.h"
#include "FaceCrop.h"
#include "Descriptor.h"

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
	SvmRegcognition();
	virtual string Recognize(cv::Mat& mat);
	virtual void defineFace();
private:
	FaceCrop face;
	Descriptor ex;
	std::vector<string> name;
	std::vector<string> all_test;
	std::vector<pfunct_type> all_pairs;
	std::vector<matrix<rgb_pixel>> faces;
	std::vector<matrix<rgb_pixel>> unknown_faces;
	std::vector<matrix<float, 0, 1>> unknown_des;
};

