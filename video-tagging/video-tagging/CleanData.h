#pragma once

using namespace std;
using namespace dlib;

class CleanData
{
public:
	CleanData() {}
	virtual void clean(string path);
private:
};