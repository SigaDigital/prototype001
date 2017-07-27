#include "stdafx.h"
#include "CleanData.h"
#include "Descriptor.h"
#include "FaceCrop.h"
#include "Manage.h"


using namespace dlib;
using namespace manage;
using namespace std;

void CleanData::Clean(string path)
{
	std::vector<string> all_file_path = get_all_file(path.c_str());
}

