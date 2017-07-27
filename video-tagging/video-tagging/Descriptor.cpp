#include "stdafx.h"
#include "Descriptor.h"
#include "Manage.h"


using namespace dlib;
using namespace manage;
using namespace std;


Descriptor::Descriptor()
{	
	current_path = get_current();
	deserialize(current_path + "/Resource/dlib_face_recognition_resnet_model_v1.dat") >> net;
}

matrix<float, 0, 1> Descriptor::GetDescriptor(matrix<rgb_pixel> face)
{
	std::vector<matrix<rgb_pixel>> faces;
	faces.push_back(face);
	std::vector<matrix<float, 0, 1>> des = net(faces);
	return  des[0];
}