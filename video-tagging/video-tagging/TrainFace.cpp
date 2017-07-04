#include "stdafx.h"
#include "TrainFace.h"
#include "Manage.h"
#include <algorithm>
#include <dlib/svm_threaded.h>
using namespace std;
using namespace dlib;

typedef matrix<double, 0, 1> sample_type;
typedef radial_basis_kernel<sample_type> kernel_type;

TrainFace::TrainFace()
{
	name.assign("");
	path = Manage::get_current();
	path = Manage::change_out(path, 2);
	
	path_des = path + "/description/";
	path_data = path + "/trainer/data/";
	path += "/trainer/name.dat";
	try { deserialize(path) >> listName; }
	catch (exception &e)
	{
		cout << e.what() << endl;
	}
}

void TrainFace::addFace(matrix<float, 0, 1> face)
{
	list.push_back(face);
}

bool TrainFace::setName(string str)
{
	found = false;
	name.assign(str);
	if (!str.empty())
	{
		for (int i = 0; i < listName.size(); i++)
			if (!str.compare(listName[i]))
				found = true;
	}
	return !found;
}

bool TrainFace::train()
{
	listName.push_back(name);
	std::vector<matrix<float, 0, 1>> tmp;
	int in = findNameIndex();
	char number[10];
	sprintf(number, "%04d", in);
	string index(number);
	string path_file = path_des; 
	
	if (!name.empty())
	{		
		if (found)
		{
			
			deserialize(path_des + index + ".dat") >> tmp;
			tmp.insert(tmp.end(), list.begin(), list.end());
			serialize(path_des + index + ".dat") << tmp;
		}
		else 
		{
			serialize(path_des + index + ".dat") << list;
		}
		serialize(path) << listName;
		
		std::vector<sample_type> samples, tmp;
		std::vector<double> labels;
		deserialize(path_des + index + ".dat") >> samples;
		std::vector<string> all_des = Manage::get_all_file(path_des.c_str());
		for (int i = 0; i < all_des.size(); i++)
		{
			if (i != in)
			{
				for (int j = 0; j < samples.size(); j++)
					labels.push_back(+1);
				deserialize(all_des[i]) >> tmp;
				for (int j = 0; j < tmp.size(); j++)
					labels.push_back(-1);
				samples.insert(samples.end(), tmp.begin(), tmp.end());

				vector_normalizer<sample_type> normalizer;
				normalizer.train(samples);
				for (unsigned long j = 0; j < samples.size(); ++j)
					samples[j] = normalizer(samples[j]);
				randomize_samples(samples, labels);

				svm_nu_trainer<kernel_type> trainer;

				trainer.set_kernel(kernel_type(0.00625));
				trainer.set_nu(0.00625);

				typedef probabilistic_decision_function<kernel_type> probabilistic_funct_type;
				typedef normalized_function<probabilistic_funct_type> pfunct_type;

				pfunct_type learned_pfunct;
				learned_pfunct.normalizer = normalizer;
				learned_pfunct.function = train_probabilistic_decision_function(trainer, samples, labels, 3);

				if(in < i)
					serialize(path_data + cast_to_string(in) + "&" + cast_to_string(i) + ".dat") << learned_pfunct;
				else
					serialize(path_data + cast_to_string(i) + "&" + cast_to_string(in) + ".dat") << learned_pfunct;
			}
		}

		return true;
	}
	return false;
}
int TrainFace::size()
{
	return list.size();
}

int TrainFace::findNameIndex()
{
	int index;
	for (int i = 0; i < listName.size(); i++)
	{
		if (listName[i].compare(name) == 0)
		{
			index = i;
			break;
		}
	}
	return index;
}