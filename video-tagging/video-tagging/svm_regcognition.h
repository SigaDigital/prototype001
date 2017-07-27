#pragma once
#include <dlib/clustering.h>
#include "Descriptor.h"
#include "FaceCrop.h"
#include "face-recognition.h"

typedef dlib::matrix<float, 0, 1> sample_type;
typedef dlib::radial_basis_kernel<sample_type> kernel_type;
typedef dlib::probabilistic_decision_function<kernel_type> probabilistic_funct_type;
typedef dlib::normalized_function<probabilistic_funct_type> pfunct_type;

class SvmRegcognition : public FaceRecognition
{
public:
	SvmRegcognition(std::string descriptor_path);
	SvmRegcognition(std::string inst_id, std::string tap_id, std::string descriptor_path);	
	virtual void Clustering();
	virtual std::string CompareFace(std::string firstPath, std::string secondPath);
	virtual std::string ImgRecognize(std::string img_file_path);
	virtual std::string Recognize(cv::Mat& mat);	
private:
	virtual double GetProbClosed(int index, dlib::matrix<float, 0, 1> des);
	virtual int CheckFace(dlib::matrix<float, 0, 1> des, double *prob, int *count);
	
	FaceCrop face;
	Descriptor ex;
	int number_of_face;
	int pre_cluster_size;
	std::string app_data_path;
	std::string data_path;
	std::string face_path;
	std::string inst_id_path;
	std::string inst_path;
	std::string pre_cluster_path;
	std::string svm_path;
	std::string tap_id_path;
	std::string unknowns_path;
	std::vector<pfunct_type> all_pairs;
	std::vector<dlib::sample_pair> edges;
	std::vector<dlib::matrix<float, 0, 1>> pre_cluster;
	std::vector<dlib::matrix<float, 0, 1>> unknown_descriptors;
	std::vector<dlib::matrix<dlib::rgb_pixel>> faces;
	std::vector<dlib::matrix<dlib::rgb_pixel>> unknown_faces;
	std::vector<std::string> list_name;
	std::vector<std::string> svm_set;	
};

