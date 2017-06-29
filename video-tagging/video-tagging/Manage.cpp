#include "stdafx.h"
#include "Manage.h"
using namespace dlib;
using namespace std;

bool Manage::ListFiles(wstring path, wstring mask, std::vector<wstring>& files) {
	HANDLE hFind = INVALID_HANDLE_VALUE;
	WIN32_FIND_DATA ffd;
	wstring spec;
	stack<wstring> directories;

	directories.push(path);
	files.clear();

	while (!directories.empty()) {
		path = directories.top();
		spec = path + L"/" + mask;
		directories.pop();

		hFind = FindFirstFile(spec.c_str(), &ffd);
		if (hFind == INVALID_HANDLE_VALUE) {
			return false;
		}

		do {
			if (wcscmp(ffd.cFileName, L".") != 0 &&
				wcscmp(ffd.cFileName, L"..") != 0) {
				if (ffd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) {
					directories.push(path + L"/" + ffd.cFileName);
				}
				else {
					files.push_back(path + L"/" + ffd.cFileName);
				}
			}
		} while (FindNextFile(hFind, &ffd) != 0);

		if (GetLastError() != ERROR_NO_MORE_FILES) {
			FindClose(hFind);
			return false;
		}

		FindClose(hFind);
		hFind = INVALID_HANDLE_VALUE;
	}

	return true;
}

string Manage::get_current()
{
	wchar_t buffer[MAX_PATH];
	GetModuleFileName(NULL, buffer, MAX_PATH);
	wstring ws(buffer);
	string path(ws.begin(), ws.end());
	return path.substr(0, path.find_last_of("\\/") + 1);
}

std::vector<string> Manage::get_all_file(const char* path)
{
	std::vector<string> path_files;
	std::vector<wstring> files;
	const size_t size = strlen(path) + 1;
	wchar_t* path_source = new wchar_t[size];
	mbstowcs(path_source, path, size);
	if (Manage::ListFiles(path_source, L"*", files))
	{
		for (std::vector<wstring>::iterator it = files.begin();it != files.end(); ++it)
		{
			string path_file_img(it->begin(), it->end());
			path_files.push_back(path_file_img);
		}
	}
	return path_files;
}

string Manage::get_name(string path)
{
	string tmp = path.substr(path.find_last_of("\\/") + 1);
	tmp = tmp.substr(0, tmp.find_last_of("."));
	return tmp;
}