#pragma once

using namespace std;
using namespace dlib;

class TrainFace
{
public:
	TrainFace();
	void addFace(matrix<float, 0, 1> face);
	bool setName(string str);
	bool train();
	int size();
private:
	int findNameIndex();
	bool found;
	string name;
	string path;
	string path_des;
	string path_data;
	std::vector<string> listName;
	std::vector<matrix<float, 0, 1>> list;
};