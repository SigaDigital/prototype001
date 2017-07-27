#pragma once

class TrainFace
{
public:
	TrainFace(std::string assigned_name, std::string trained_path, std::string descriptor_path, double gamma_in, double nu_in);
	void Train();
private:
	bool SetName(std::string str);
	int FindNameIndex();

	bool found;	
	double gamma_value;
	double nu_value;
	std::string app_data_path;
	std::string data_path;
	std::string faces_path;
	std::string listName_file_path;
	std::string name_descriptor;
	std::string svm_path;
	std::string unknowns_path;
	std::string unknown_src_des_path;
	std::string unknown_src_path;
	std::vector<dlib::matrix<float, 0, 1>> unknown_des;	
	std::vector<std::string> list_name;	
};