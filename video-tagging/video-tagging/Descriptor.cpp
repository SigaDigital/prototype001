#include "stdafx.h"
#include "Descriptor.h"
#include "Manage.h"

using namespace dlib;
using namespace std;

Descriptor::Descriptor()
{	
	string path = Manage::get_current();
	path = Manage::change_out(path, 2);
	deserialize(path + "/resource/dlib_face_recognition_resnet_model_v1.dat") >> net;
}

matrix<float, 0, 1> Descriptor::get_description(matrix<rgb_pixel> face)
{
	std::vector<matrix<rgb_pixel>> faces;
	faces.push_back(face);
	std::vector<matrix<float, 0, 1>> des = net(faces);
	return  des[0];
}