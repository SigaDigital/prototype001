#include "stdafx.h"
#include "Manage.h"
#include "FaceCrop.h"
#include <dlib/image_io.h>

using namespace dlib;
using namespace std;
using namespace manage;

FaceCrop::FaceCrop()
{
	current_path = get_current();
	
	deserialize(current_path + "/Resource/shape_predictor_68_face_landmarks.dat") >> sp;
	detector = get_frontal_face_detector();		
}

//Get all face images form matrix
std::vector<matrix<rgb_pixel>>  FaceCrop::GetFace(cv::Mat& frame)
{
	std::vector<matrix<rgb_pixel>> faces;
	imwrite(current_path + "/tmp.jpg", frame);	
	cv_image<rgb_pixel> tmp(frame);
	matrix<rgb_pixel> cimg;
	load_image(cimg, current_path + "/tmp.jpg");
	pyramid_up(cimg);

	for (auto face : detector(cimg))
	{
		array2d<rgb_pixel> img2d;
		auto shape = sp(cimg, face);
		matrix<rgb_pixel> face_chip;

		extract_image_chip(cimg, get_face_chip_details(shape, 150, 0.25), face_chip);
		faces.push_back(move(face_chip));				
	}
	return faces;
}

//Get all face images form image file path
std::vector<matrix<rgb_pixel>>  FaceCrop::GetFace(string img_file_path)
{
	std::vector<matrix<rgb_pixel>> faces;
	matrix<rgb_pixel> cimg;
	load_image(cimg, img_file_path);

	for (auto face : detector(cimg))
	{
		array2d<rgb_pixel> img2d;
		auto shape = sp(cimg, face);
		matrix<rgb_pixel> face_chip;

		extract_image_chip(cimg, get_face_chip_details(shape, 150, 0.25), face_chip);
		faces.push_back(move(face_chip));
	}
	return faces;
}