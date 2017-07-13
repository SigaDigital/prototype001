#include "stdafx.h"
#include "Manage.h"
#include "FaceCrop.h"
#include <dlib/image_io.h>

using namespace dlib;
using namespace std;

FaceCrop::FaceCrop()
{
	currentPath = Manage::get_current();
	
	deserialize(currentPath + "/Resource/shape_predictor_68_face_landmarks.dat") >> sp;
	detector = get_frontal_face_detector();		
}

std::vector<matrix<rgb_pixel>>  FaceCrop::get_face(cv::Mat& frame)
{
	std::vector<matrix<rgb_pixel>> faces;
	imwrite(currentPath + "tmp.jpg", frame);	
	cv_image<rgb_pixel> tmp(frame);
	matrix<rgb_pixel> cimg;
	load_image(cimg, currentPath + "tmp.jpg");

	for (auto face : detector(cimg))
	{
		array2d<rgb_pixel> img2d;
		char number[10];
		auto shape = sp(cimg, face);
		matrix<rgb_pixel> face_chip;

		extract_image_chip(cimg, get_face_chip_details(shape, 150, 0.25), face_chip);
		faces.push_back(move(face_chip));				
	}
	return faces;
}