#include "stdafx.h"
#include "Manage.h"
#include "FaceCrop.h"
#include <dlib/image_io.h>

using namespace dlib;
using namespace std;

FaceCrop::FaceCrop()
{
	string path = Manage::get_current();
	path = Manage::change_out(path, 2);
	detector = get_frontal_face_detector();
	deserialize(path + "/resource/shape_predictor_68_face_landmarks.dat") >> sp;
	total = 0;
	
}

std::vector<matrix<rgb_pixel>>  FaceCrop::get_face(cv::Mat& frame)
{
	string path = Manage::get_current();
	imwrite(path + "tmp.jpg", frame);
	std::vector<matrix<rgb_pixel>> faces;
	cv_image<rgb_pixel> tmp(frame);
	matrix<rgb_pixel> cimg;
	load_image(cimg, path + "tmp.jpg");
	for (auto face : detector(cimg))
	{
		array2d<rgb_pixel> img2d;
		char number[10];
		auto shape = sp(cimg, face);
		matrix<rgb_pixel> face_chip;
		extract_image_chip(cimg, get_face_chip_details(shape, 150, 0.25), face_chip);
		faces.push_back(move(face_chip));
		assign_image(img2d, faces.back());
		sprintf(number, "%04d", total++);
		save_jpeg(img2d, path +  + "/tmp_crop_face/tmp_" + string(number) +".jpg");
	}
	return faces;
}