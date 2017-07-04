#include "stdafx.h"
#include "Manage.h"
#include "FaceCrop.h"

using namespace dlib;
using namespace std;

FaceCrop::FaceCrop()
{
	string path = Manage::get_current();
	path = Manage::change_out(path, 2);
	detector = get_frontal_face_detector();
	deserialize(path + "/resource/shape_predictor_68_face_landmarks.dat") >> sp;
}

std::vector<matrix<rgb_pixel>>  FaceCrop::get_face(cv::Mat& frame)
{
	std::vector<matrix<rgb_pixel>> faces;
	cv_image<rgb_pixel> cimg(frame);

	for (auto face : detector(cimg))
	{
		auto shape = sp(cimg, face);
		matrix<rgb_pixel> face_chip;
		extract_image_chip(cimg, get_face_chip_details(shape, 150, 0.25), face_chip);
		faces.push_back(move(face_chip));
	}
	return faces;
}