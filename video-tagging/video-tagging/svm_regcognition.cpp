#include "stdafx.h"
#include "svm_regcognition.h"
#include "TrainFace.h"
#include "Manage.h"
#include <algorithm> 
#include <direct.h>
#include <fstream>

using namespace dlib;
using namespace std;

SvmRegcognition::SvmRegcognition()
{
	string path = Manage::get_current();
	mkdir((path + "/tmp_crop_face").c_str());
	outfile.open(path + "/tmp_crop_face/info.txt", std::ios_base::app);
	path = Manage::change_out(path, 2);
	string train_path = path + "/trainer/data/";
	all_test = Manage::get_all_file(train_path.c_str());
	path += "/trainer/name.dat";
	{
		deserialize(path) >> name;
	}


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
	cout << "star recognize ..." << endl;
	string output = "";
	std::vector<matrix<rgb_pixel>> faces = face.get_face(mat);
	matrix<float, 0, 1> test;
	
	for (int i = 0; i < faces.size(); i++)
	{
		test = ex.get_description(faces[i]);
		double *prob = new double[name.size()];
		int *count = new int[name.size()];
		memset(prob, 0, sizeof(double) * name.size());
		memset(count, 0, sizeof(int) * name.size());
		int index = 0;
		double max = 0;
		const double invSize = 1.0 / (name.size() - 1) ;
		
		for (int i = 0; i < all_test.size() - 1; i++)
		{
			pfunct_type learned_pfunct = all_pairs[i];

			string data_name = Manage::get_name(all_test[i]);
			string nameA = data_name.substr(0, data_name.find_first_of("&"));
			string nameB = data_name.substr(data_name.find_last_of("&") + 1);

			double dis = learned_pfunct(test);
			
			int indexA = atoi(nameA.c_str());
			int indexB = atoi(nameB.c_str());
			if (dis > 0.5)
				prob[indexA] += (dis - 0.5) * 2 * invSize;
			else
				prob[indexB] += (1 - dis - 0.5) * 2 * invSize;
				
			if(dis > 0.8)
				count[indexA]++;
			else if(dis < 0.2)
				count[indexB]++;

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
		if (prob[index] < 0.75)
		{
			unknown_faces.push_back(faces[i]);		
			unknown_des.push_back(test);
			
		}
		if(prob[index] < 0.75)
			outfile << "unknown"  << " " << name[index] <<endl;
		else
			outfile << "known" << " " << name[index] << endl;
		
		output.append(name[index] + "---> confidence: " + cast_to_string(prob[index]) +
						" Score:" + cast_to_string(count[index]) + "/" + cast_to_string(name.size()-1));
		delete[] prob;
	}

	return output;
}

void SvmRegcognition::clustering()
{
	std::vector<matrix<float, 0, 1>> des_tmp;
	std::vector<sample_pair> edges;
	for (size_t i = 0; i < unknown_des.size(); ++i)
	{
		for (size_t j = i + 1; j < unknown_des.size(); ++j)
		{
			if (length(unknown_des[i] - unknown_des[j]) < 0.5)
				edges.push_back(sample_pair(i, j));
		}
	}
	std::vector<unsigned long> labels;
	
	const auto num_clusters = chinese_whispers(edges, labels, 200);
	
	for (size_t cluster_id = 0; cluster_id < num_clusters; ++cluster_id)
	{
		TrainFace unknown;
		image_window win_cluster;
		std::vector<matrix<rgb_pixel>> temp;
		for (size_t j = 0; j < labels.size(); ++j)
		{
			if (cluster_id == labels[j])
				temp.push_back(unknown_faces[j]);		
			//unknown.addFace(unknown_des[j]);
		}
		win_cluster.set_title("face cluster " + cast_to_string(cluster_id));
		win_cluster.set_image(tile_images(temp));
		//std::getline(std::cin, tmpName);
		//if (!tmpName.empty())
		{			
			//unknown.setName(tmpName);
			//unknown.train();
		}
		cin.get();
	}
}
