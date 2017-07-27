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
using namespace manage;

//Constructor for video regcognition
SvmRegcognition::SvmRegcognition(string inst_id, string tap_id, string descriptor_path)
{	
	//String path
	app_data_path = string(getenv("APPDATA")) + "/VideoTagging";
	data_path = app_data_path + "/Data";
	unknowns_path = data_path + "/Unknows";

	inst_path = app_data_path + "/Instance";
	inst_id_path = inst_path + "/" + inst_id;
	tap_id_path = inst_id_path + "/" + tap_id;

	face_path = descriptor_path + "/Faces";
	svm_path = descriptor_path + "/Svms";

	//Create Directory
	mkdir(app_data_path.c_str());
	mkdir(inst_path.c_str());
	mkdir(inst_id_path.c_str());
	mkdir(tap_id_path.c_str());
	mkdir(data_path.c_str());
	mkdir(unknowns_path.c_str());
	
	svm_set = get_all_file(svm_path);

	//Load name list
	string listName_file_path = descriptor_path + "/Name.dat";
	deserialize(listName_file_path) >> list_name;

	number_of_face = 0;
	pre_cluster_size = 0;

	//Pre-Cluster is using for clustring unknown face and Group previous face and recent face that are similar together
	pre_cluster_path = unknowns_path + "/PreCluster.dat";
	if (is_file_exist(pre_cluster_path))
	{
		deserialize(pre_cluster_path) >> pre_cluster;
		pre_cluster_size = pre_cluster.size();
	}

	//Load all test set from Svms
	for (int i = 0; i < svm_set.size(); i++)
	{
		pfunct_type learned_pfunct;
		deserialize(svm_set[i]) >> learned_pfunct;
		all_pairs.push_back(learned_pfunct);
	}
}

//Constructor for image testing
SvmRegcognition::SvmRegcognition(string descriptor_path)
{
	//Load name list
	string listName_file_path = descriptor_path + "/Name.dat";
	deserialize(listName_file_path) >> list_name;

	svm_path = descriptor_path + "/Svms";
	svm_set = get_all_file(svm_path);

	//Load all test set
	for (int i = 0; i < svm_set.size(); i++)
	{
		pfunct_type learned_pfunct;
		deserialize(svm_set[i]) >> learned_pfunct;
		all_pairs.push_back(learned_pfunct);
	}
}

string SvmRegcognition::ImgRecognize(string img_file_path)
{
	string output = "";
	std::vector<matrix<rgb_pixel>> faces = face.GetFace(img_file_path);
	matrix<float, 0, 1> des;

	//Processing each face
	for (int i = 0; i < faces.size(); i++)
	{
		int index = 0;
		double *prob = new double[list_name.size()];
		int *count = new int[list_name.size()];
		memset(prob, 0, sizeof(double) * list_name.size());
		memset(count, 0, sizeof(int) * list_name.size());

		des = ex.GetDescriptor(faces[i]);
		index = CheckFace(des, prob, count);

		//Determine the confidence
		double prob_closed = GetProbClosed(index, des);
		const double ACCEPTED_PROB = 0.8;
		const double CONNECTED_ACCEPTED_PROB = 0.5;
		//Known
		if (
			(double)count[index] / list_name.size() > ACCEPTED_PROB
			&& prob[index] > ACCEPTED_PROB
			&& prob_closed > CONNECTED_ACCEPTED_PROB
			)
		{
			output.append("UNKNOWN");
		}
		else //Unknown
		{
			output.append(list_name[index]);
		}


		delete[] prob, count;
	}

	return output;
}

string SvmRegcognition::Recognize(cv::Mat& mat)
 {
	string output = "";
	std::vector<matrix<rgb_pixel>> faces = face.GetFace(mat);
	matrix<float, 0, 1> des;
	
	//Processing each face
	for (int i = 0; i < faces.size(); i++)
	{		
		int index = 0;
		double *prob = new double[list_name.size()];
		int *count = new int[list_name.size()];
		memset(prob, 0, sizeof(double) * list_name.size());
		memset(count, 0, sizeof(int) * list_name.size());

		des = ex.GetDescriptor(faces[i]);
		index = CheckFace(des, prob, count);		
		
		//Determine the confidence
		double prob_closed = GetProbClosed(index, des);
		const double ACCEPTED_PROB = 0.8;
		const double CONNECTED_ACCEPTED_PROB = 0.5;
		//Known
		if(
			(double)count[index] / list_name.size() > ACCEPTED_PROB
			&& prob[index] > ACCEPTED_PROB
			&& prob_closed > CONNECTED_ACCEPTED_PROB
			) 
		{
			string dst_known_path = tap_id_path + "/" + list_name[index];
			array2d<rgb_pixel> img2d;
			char number[10];
			sprintf(number, "%04d", number_of_face++);
			string known_image_file_path = dst_known_path + "/" + list_name[index] + "_" + string(number) + ".jpg";

			mkdir(dst_known_path.c_str());

			assign_image(img2d, faces[i]);
			save_jpeg(img2d, known_image_file_path);
		}		
		else //Unknown
		{
			pre_cluster.push_back(des);
			if (pre_cluster.size() < 2) // avoid chinese whisper problem
				pre_cluster.push_back(des);

			unknown_faces.push_back(faces[i]);
			unknown_descriptors.push_back(des);
		}

		output.append(list_name[index] + "---> confidence: " + cast_to_string(prob[index]) +
						" Score:" + cast_to_string(count[index]) + "/" + cast_to_string(list_name.size()-1));
		number_of_face++;
		delete[] prob, count;
	}
	
	return output;
}

//Compare distance between 2 face images
string SvmRegcognition::CompareFace(string firstPath, string secondPath)
{
	std::vector<matrix<rgb_pixel>> first = face.GetFace(firstPath);
	std::vector<matrix<rgb_pixel>> second = face.GetFace(secondPath);

	matrix<float, 0, 1> descriptor_first, descriptor_second;
	descriptor_first = ex.GetDescriptor(first[0]);
	descriptor_second = ex.GetDescriptor(second[0]);

	return cast_to_string(length(descriptor_first - descriptor_second));
}

int SvmRegcognition::CheckFace(matrix<float, 0, 1> des, double *prob, int *count)
{
	double max = 0;
	int index = 0;
	const double invert_size = 1.0 / (list_name.size() - 1);

	//Check all test set from Svms
	for (int j = 0; j < svm_set.size(); j++)
	{
		pfunct_type learned_pfunct = all_pairs[j];

		string data_name = get_name(svm_set[j]);
		string nameA = data_name.substr(0, data_name.find_first_of("&"));
		string nameB = data_name.substr(data_name.find_last_of("&") + 1);

		double dis = learned_pfunct(des);

		int index_first = atoi(nameA.c_str());
		int index_second = atoi(nameB.c_str());

		const double MID_PROB = 0.5;
		if (dis > MID_PROB)
		{
			prob[index_first] += (dis - MID_PROB) * 2 * invert_size;
		}
		else
		{
			prob[index_second] += (1 - dis - MID_PROB) * 2 * invert_size;
		}

		const double ACCEPTED_VALUE = 0.2;
		//Score counting
		if (dis > (1 - ACCEPTED_VALUE))
		{
			count[index_first]++;
		}
		else if (dis < ACCEPTED_VALUE)
		{
			count[index_second]++;
		}

		//Check closest face
		if (max < prob[index_first])
		{
			index = index_first;
			max = prob[index_first];
		}
		if (max < prob[index_second])
		{
			index = index_second;
			max = prob[index_second];
		}
	}

	return index;
}


void SvmRegcognition::Clustering()
{
	const double ACCEPTED_DISTANCE = 0.45;

	//create a grapth of connected faces
	for (size_t i = 0; i < pre_cluster.size(); ++i)
	{
		for (size_t j = i + 1; j < pre_cluster.size(); ++j)
		{
			if (length(pre_cluster[i] - pre_cluster[j]) < ACCEPTED_DISTANCE)
				edges.push_back(sample_pair(i, j));
		}
	}
	
	//Use the chines whispers grapth clustring
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
		if (is_file_exist(unknow_des_files_path))
		{
			deserialize(unknow_des_files_path) >> unknown_des;
		}

		for (size_t j = pre_cluster_size; j < labels.size(); ++j)
		{
			//Group the similar image 
			if (cluster_id == labels[j])
			{				
				array2d<rgb_pixel> img2d;
				char number[10];
				int id = unknown_des.size();
				sprintf(number, "%04d", id);
				string unknown_image_file_path = dst_unknown_path + "/" + unknown_name + "_" + string(number) + ".jpg";

				assign_image(img2d, unknown_faces[j - pre_cluster_size]);
				save_jpeg(img2d, unknown_image_file_path);

				unknown_des.push_back(unknown_descriptors[j - pre_cluster_size]);
			}
		}		
		serialize(unknow_des_files_path) << unknown_des;
	}
	serialize(pre_cluster_path) << pre_cluster;
}

double SvmRegcognition::GetProbClosed(int index, matrix<float, 0, 1> des)
{
	//check distance
	char number_index[10];
	int closed_face = 0;
	std::vector<dlib::matrix<float, 0, 1>> check_face;
	sprintf(number_index, "%04d", index);
	deserialize(face_path + "/" + string(number_index) + ".dat") >> check_face;

	const double ACCEPTED_DISTANCE = 0.5;
	for (int j = 0; j < check_face.size(); j++)
	{
		if (length(des - check_face[j]) < ACCEPTED_DISTANCE)
		{
			closed_face++;
		}
	}

	return (double)closed_face / check_face.size();
}