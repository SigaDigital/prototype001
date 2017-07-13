#include "stdafx.h"
#include "svm_regcognition.h"
#include "TrainFace.h"
#include "Manage.h"
#include <algorithm> 
#include <direct.h>
#include <fstream>
#include <dlib/image_io.h>
#include <stdio.h>

using namespace dlib;
using namespace std;

SvmRegcognition::SvmRegcognition(string inst_id, string tap_id, string descriptor_path)
{	
	app_data_path = string(getenv("APPDATA")) + "/VideoTagging";
	data_path = app_data_path + "/Data";
	unknowns_path = data_path + "/Unknows";

	inst_path = app_data_path + "/Instance";
	inst_id_path = inst_path + "/" + inst_id;
	tap_id_path = inst_id_path + "/" + tap_id;

	svm_path = descriptor_path + "/Svms";

	mkdir(app_data_path.c_str());
	mkdir(inst_path.c_str());
	mkdir(inst_id_path.c_str());
	mkdir(tap_id_path.c_str());
	mkdir(data_path.c_str());
	mkdir(unknowns_path.c_str());
	
	svmSet = Manage::get_all_file(svm_path.c_str());

	string listName_file_path = descriptor_path + "/Name.dat";
	deserialize(listName_file_path) >> listName;

	numberFace = 0;
	preCluster_Size = 0;

	preCluster_Path = unknowns_path + "/PreCluster.dat";
	if (Manage::isFileExist(preCluster_Path.c_str()))
	{
		deserialize(preCluster_Path) >> preCluster;
		preCluster_Size = preCluster.size();
	}

	//Load all test set
	for (int i = 0; i < svmSet.size(); i++)
	{
		pfunct_type learned_pfunct;
		deserialize(svmSet[i]) >> learned_pfunct;
		all_pairs.push_back(learned_pfunct);
	}
}

SvmRegcognition::~SvmRegcognition()
{
	serialize(preCluster_Path) << preCluster;
}

string SvmRegcognition::Recognize(cv::Mat& mat)
{
	cout << "star recognize ..." << endl;
	string output = "";
	std::vector<matrix<rgb_pixel>> faces = face.get_face(mat);
	matrix<float, 0, 1> des;
	
	//Processing each face
	for (int i = 0; i < faces.size(); i++)
	{		
		int index = 0;
		double max = 0;
		const double invSize = 1.0 / (listName.size() - 1) ;

		double *prob = new double[listName.size()];
		int *count = new int[listName.size()];
		memset(prob, 0, sizeof(double) * listName.size());
		memset(count, 0, sizeof(int) * listName.size());
		
		des = ex.get_descriptor(faces[i]);

		//Check all test set
		for (int j = 0; j < svmSet.size(); j++)
		{
			pfunct_type learned_pfunct = all_pairs[j];

			string data_name = Manage::get_name(svmSet[j]);
			string nameA = data_name.substr(0, data_name.find_first_of("&"));
			string nameB = data_name.substr(data_name.find_last_of("&") + 1);

			double dis = learned_pfunct(des);
			
			int indexA = atoi(nameA.c_str());
			int indexB = atoi(nameB.c_str());

			if (dis > 0.5)
				prob[indexA] += (dis - 0.5) * 2 * invSize;
			else
				prob[indexB] += (1 - dis - 0.5) * 2 * invSize;
				
			//Score counting
			if(dis > 0.8)
				count[indexA]++;
			else if(dis < 0.2)
				count[indexB]++;

			//Check nearest face
			if (max < prob[indexA])
			{
				index = indexA;
				max = prob[indexA];
			}
			if (max < prob[indexB])
			{
				index = indexB;
				max = prob[indexB];
			}
		}			

		//Determine the confidence
		if (prob[index] < 0.8)	//Unknown
		{
			preCluster.push_back(des);

			unknown_faces.push_back(faces[i]);
			unknown_descriptors.push_back(des);
		}	
		else  //Known
		{
			string dst_known_path = tap_id_path + "/" + listName[index];
			array2d<rgb_pixel> img2d;
			char number[10];
			sprintf(number, "%04d", Manage::number_of_files(dst_known_path));
			string known_image_file_path = dst_known_path + "/" + listName[index] + "_" + string(number) + ".jpg";

			mkdir(dst_known_path.c_str());

			assign_image(img2d, faces[i]);
			save_jpeg(img2d, known_image_file_path);
		}					

		output.append(listName[index] + "---> confidence: " + cast_to_string(prob[index]) +
						" Score:" + cast_to_string(count[index]) + "/" + cast_to_string(listName.size()-1));
		numberFace++;
		delete[] prob;
	}
	
	return output;
}

void SvmRegcognition::clustering()
{
	const double treshold = 0.5;

	for (size_t i = 0; i < preCluster.size(); ++i)
	{
		for (size_t j = i + 1; j < preCluster.size(); ++j)
		{
			if (length(preCluster[i] - preCluster[j]) < treshold)
				edges.push_back(sample_pair(i, j));
		}
	}
	
	std::vector<unsigned long> labels;	
	const auto num_clusters = chinese_whispers(edges, labels, 200);
	
	for (size_t cluster_id = 0; cluster_id < num_clusters; ++cluster_id)
	{
		char number[10];
		sprintf(number, "%04d", cluster_id);
		string unknown_name = "Unknown_" + string(number);
		string dst_unknown_path = unknowns_path + "/" + unknown_name;
		mkdir(dst_unknown_path.c_str());

		std::vector<matrix<float, 0, 1>> unknown_des;
		string unknow_des_files_path = dst_unknown_path + "/Descriptor_" + unknown_name + ".dat";
		if (Manage::isFileExist(unknow_des_files_path.c_str()))
		{
			deserialize(unknow_des_files_path) >> unknown_des;
		}

		for (size_t j = preCluster_Size; j < labels.size(); ++j)
		{
			if (cluster_id == labels[j])
			{				
				array2d<rgb_pixel> img2d;
				char number[10];
				int id = unknown_des.size();
				sprintf(number, "%04d", id);
				string unknown_image_file_path = dst_unknown_path + "/" + unknown_name + "_" + string(number) + ".jpg";

				assign_image(img2d, unknown_faces[j - preCluster_Size]);
				save_jpeg(img2d, unknown_image_file_path);

				unknown_des.push_back(unknown_descriptors[j - preCluster_Size]);
			}
		}		

		serialize(unknow_des_files_path) << unknown_des;
	}
}
