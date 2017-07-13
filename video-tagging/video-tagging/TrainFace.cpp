#include "stdafx.h"
#include "TrainFace.h"
#include "Manage.h"
#include <algorithm>
#include <dlib/svm_threaded.h>
using namespace std;
using namespace dlib;

typedef matrix<double, 0, 1> sample_type;
typedef radial_basis_kernel<sample_type> kernel_type;

TrainFace::TrainFace(string assigned_name, string trained_path, string descriptor_path)
{
	app_data_path = string(getenv("APPDATA")) + "/VideoTagging";
	data_path = app_data_path + "/Data";
	unknowns_path = data_path + "/Unknows";

	unknown_src_path = trained_path;
	unknown_src_des_path = unknown_src_path + "/Descriptor_" + Manage::get_dirName(unknown_src_path) + ".dat";

	if (Manage::isFileExist(unknown_src_des_path.c_str()))
		deserialize(unknown_src_des_path) >> unknown_des;
	
	svm_path = descriptor_path + "/Svms";
	faces_path = descriptor_path + "/Faces";

	listName_file_path = descriptor_path + "/Name.dat";
	if (Manage::isFileExist(listName_file_path.c_str()))
		deserialize(listName_file_path) >> listName;
	setName(assigned_name);
}

bool TrainFace::setName(string str)
{
	found = false;
	name_descriptor.assign(str);
	if (!str.empty())
	{
		for (int i = 0; i < listName.size(); i++)
			if (!str.compare(listName[i]))
				found = true;
	}
	return !found;
}

void TrainFace::train()
{	
	if(!found)
		listName.push_back(name_descriptor);
	
	int index_name = findNameIndex();
	char number[10];

	sprintf(number, "%04d", index_name);
	string index(number);
	string descriptor_path_file = faces_path + "/" + index + ".dat"; 	
	
	std::vector<matrix<float, 0, 1>> descriptor;
	if (found)
		deserialize(descriptor_path_file) >> descriptor;
				
	serialize(listName_file_path) << listName;
	descriptor.insert(descriptor.end(), unknown_des.begin(), unknown_des.end());
	serialize(descriptor_path_file) << descriptor;

	
	std::vector<sample_type> samples, tmp;
	std::vector<double> labels;
	deserialize(descriptor_path_file) >> samples;
	std::vector<string> all_des = Manage::get_all_file(faces_path.c_str());
	for (int i = 0; i < all_des.size(); i++)
	{
		if (i != index_name)
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
						
			serialize(svm_path + "/" + cast_to_string(index_name) + "&" + cast_to_string(i) + ".dat") << learned_pfunct;
		}
	}
}

int TrainFace::findNameIndex()
{
	int index;
	for (int i = 0; i < listName.size(); i++)
	{
		if (listName[i].compare(name_descriptor) == 0)
		{
			index = i;
			break;
		}
	}
	return index;
}