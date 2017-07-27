#pragma once
#include <dlib/dnn.h>

class FaceCrop
{
public:
	FaceCrop();
	std::vector<dlib::matrix<dlib::rgb_pixel>>  FaceCrop::GetFace(cv::Mat& frame);
	std::vector<dlib::matrix<dlib::rgb_pixel>>  FaceCrop::GetFace(std::string img_file_path);
private:
	dlib::frontal_face_detector detector;
	dlib::shape_predictor sp;
	std::string current_path;
};