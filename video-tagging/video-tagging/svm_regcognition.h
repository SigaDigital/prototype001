#pragma once
#include <dlib/clustering.h>
#include "face-recognition.h"
#include "FaceCrop.h"
#include "Descriptor.h"

using namespace dlib;
using namespace std;

typedef matrix<float, 0, 1> sample_type;
typedef radial_basis_kernel<sample_type> kernel_type;
typedef probabilistic_decision_function<kernel_type> probabilistic_funct_type;
typedef normalized_function<probabilistic_funct_type> pfunct_type;

class SvmRegcognition: public FaceRecognition
{
public:
	SvmRegcognition(string inst_id, string tap_id, string descriptor_path);
	virtual string Recognize(cv::Mat& mat);
	virtual void clustering(); 
	virtual ~SvmRegcognition();
private:
	FaceCrop face;
	Descriptor ex;
	std::vector<string> listName;
	std::vector<string> svmSet;
	std::vector<pfunct_type> all_pairs;
	std::vector<matrix<rgb_pixel>> faces;
	std::vector<matrix<rgb_pixel>> unknown_faces;
	std::vector<matrix<float, 0, 1>> unknown_descriptors;
	std::vector<sample_pair> edges;
	std::vector<matrix<float, 0, 1>> preCluster;
	string app_data_path;	
	string data_path;
	string unknowns_path;
	string inst_path;
	string inst_id_path;
	string tap_id_path;
	string svm_path;
	string preCluster_Path;
	int numberFace;	
	int preCluster_Size;
};

