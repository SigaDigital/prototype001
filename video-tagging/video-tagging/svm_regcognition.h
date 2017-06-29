#pragma once

using namespace dlib;
using namespace std;

typedef matrix<float, 0, 1> sample_type;
typedef radial_basis_kernel<sample_type> kernel_type;

class SvmRegcognition
{
public:
	SvmRegcognition(char* path);
	void SvmRegcognition::SvmRegcognizer(matrix<float, 0, 1>);
private:
	string path;
    string train_path;
	std::vector<string> name;
	std::vector<string> all_test;
};

