#pragma once

using namespace std;
using namespace dlib;

class TrainFace
{
public:
	TrainFace(string assigned_name, string trained_path, string descriptor_path, double gamma_in, double nu_in);
	void train();
	
private:
	int findNameIndex();
	double gammaValue;
	double nuValue;
	bool setName(string str);
	bool found;
	string name_descriptor;
	string app_data_path;
	string data_path;
	string unknowns_path;
	string unknown_src_path;
	string unknown_src_des_path;
	string svm_path;
	string faces_path;
	string listName_file_path;
	std::vector<string> listName;
	std::vector<matrix<float, 0, 1>> unknown_des;
};