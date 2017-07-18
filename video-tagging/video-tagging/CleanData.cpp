#include "stdafx.h"
#include "CleanData.h"
#include "FaceCrop.h"
#include "Descriptor.h"
#include "Manage.h"
using namespace std;
using namespace dlib;

void CleanData::clean(string path)
{
	std::vector<string> all_file_path = Manage::get_all_file(path.c_str());	
}


