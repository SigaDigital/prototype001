#pragma once
#include <dlib/dnn.h>
using namespace dlib;
using namespace std;
class FaceCrop
{
public:
	FaceCrop();
	std::vector<matrix<rgb_pixel>>  FaceCrop::get_face(cv::Mat& frame);
	std::vector<matrix<rgb_pixel>>  FaceCrop::get_face(string img_file_path);
private:
	shape_predictor sp;
	frontal_face_detector detector;
	string currentPath;
};