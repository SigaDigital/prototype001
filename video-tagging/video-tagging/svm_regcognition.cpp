#include "stdafx.h"
#include "svm_regcognition.h"
#include "Descriptor.h"
#include "Manage.h"


using namespace dlib;
using namespace std;

SvmRegcognition::SvmRegcognition(char* path)
{
	all_test = Manage::get_all_file(path);
	string path_file(strcat(path, "\\name.dat"));
	deserialize(path_file) >> name;	

	for (int i = 0; i < all_test.size(); i++)
	{
		pfunct_type learned_pfunct;
		if (Manage::get_name(all_test[i]).compare("name"))
		{
			deserialize(all_test[i]) >> learned_pfunct;
			all_pairs.push_back(learned_pfunct);
		}
	}
}

string SvmRegcognition::Recognize(cv::Mat& mat)
{
	string output = "";
	Descriptor ex;
	std::vector<matrix<float, 0, 1>> tmp = ex.get_description(mat);
	for (int i = 0; i < tmp.size(); i++)
	{
		matrix<float, 0, 1> test = tmp[i];
		double *count = new double[name.size()];
		memset(count, 0, sizeof(double) * name.size());
		int index = 0;
		double max = 0;
		const double invSize = 1.0 / (name.size() - 1) ;

		for (int i = 0; i < all_test.size() - 1; i++)
		{
			pfunct_type learned_pfunct = all_pairs[i];

			string data_name = Manage::get_name(all_test[i]);
			string nameA = data_name.substr(0, data_name.find_first_of("&"));
			string nameB = data_name.substr(data_name.find_last_of("&") + 1);

			double prob = learned_pfunct(test);

			int indexA = atoi(nameA.c_str());
			int indexB = atoi(nameB.c_str());
			count[indexA] += (prob - 0.5) * 2 * invSize;
			count[indexB] += (1 - prob - 0.5) * 2 * invSize;

			if (max < count[indexA])
			{
				index = indexA;
				max = count[indexA];
			}
			if (max < count[indexB])
			{
				index = indexB;
				max = count[indexB];
			}
		}
		
		output.append(name[index] + "---> confidence: " + cast_to_string(count[index]));
		delete[] count;
	}

	return output;
}
