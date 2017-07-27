#pragma once
#include <vector>

namespace manage
{
	bool list_files(std::wstring path, std::wstring mask, std::vector<std::wstring>& files);
	bool is_file_exist(std::string fileName);
	int number_of_files(std::string path);
	void copy_file(std::string sourcePath, std::string destPath);
	std::string get_current();
	std::string get_parent_name(std::string path);
	std::string get_name(std::string path);
	std::vector<std::string> get_all_file(std::string path);
}