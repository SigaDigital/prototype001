#pragma once
using namespace std;
using namespace dlib;

class Manage
{
public:	
	static bool ListFiles(wstring path, wstring mask, std::vector<wstring>& files);
	static string get_current();
	static std::vector<string> get_all_file(const char* path);
	static string get_name(string path);
	static string change_out(string path, int n);
	static void copyFile(string SRC, string DEST);
	static bool isFileExist(const char *fileName);
	static int number_of_files(string path);
	static string get_dirName(string path);
};