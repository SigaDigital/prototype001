// video-tagging.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "frontal-face-filter.h"
#include "sampler.h"
#include "sighthound-recognition.h"
#include "svm_regcognition.h"
#include "TrainFace.h"

using namespace dlib;
using namespace std;

int main(int argc, char** argv)
{
	if (argc < 2)
	{
		cout << "Video Regcognition" << endl;
		cout << "Sighthound method" << endl;
		cout << "Usage : " << argv[0] << "rec [video file path] sighthound" << endl;
		cout << "Svm method" << endl;
		cout << "Usage : " << argv[0] << "rec [video file path] svm [descriptor directory path]" << endl;
		cout << "--------------------------------" << endl;
		cout << "\"Train new face\" usage : " << argv[0] << "  test [assigned name] [images directory path] " 
			<< "[descriptor directory path] [gamma value] [nu value]" << endl;
		cout << "\"Test image file\" usage : " << argv[0] << " [image file path] [descriptor director path]" << endl;
		cout << "\"Compare distance between 2 image files\" usage : " << argv[0] << " [1st img file path] [2nd img file path] [descriptor directory path]" << endl;
		return -1;
	}
	
	//Select Mode
	if (!strcmp(argv[1], "train"))
	{
		double gamma, nu;
		sscanf(argv[5], "%lf", &gamma);
		sscanf(argv[6], "%lf", &nu);
		TrainFace *train = new TrainFace(argv[2], argv[3], argv[4], gamma, nu);
		train->Train();
		delete train;
	}
	else if(!strcmp(argv[1], "rec"))
	{
		Sampler sampler(argv[2]);
		FrontalFaceFilter filter;
		cv::Mat frame;

		sampler.setRate(atoi(argv[4]));
		FaceRecognition* pRecognizer = 0;
		if (strcmp(argv[3], "sighthound") == 0)
		{
			pRecognizer = new SighthoundRecognition("7796KskdhG1nMlLjaTWh155dYsbeZGqJzsHq");
		}
		else
		{
			pRecognizer = new SvmRegcognition(string(argv[5]), string(argv[6]), string(argv[7]));
		}

		while (sampler.Next(frame))
		{
			//cout << "Next Frame" << endl;
			//cv::imshow("frame", frame);
			//cv::waitKey(100);
			if (filter.Exec(frame))
			{
				cout << "frame execute ..." << endl;
				std::cout << pRecognizer->Recognize(frame) << std::endl;
			}
		}
		pRecognizer->Clustering();

		delete pRecognizer;
		std::cout << "end" << endl;
	}
	else if(!strcmp(argv[1], "test"))
	{
		FaceRecognition* imgReg = new SvmRegcognition(string(argv[3]));

		std::cout << imgReg->ImgRecognize(string(argv[2]));

		delete imgReg;
	}
	else if (!strcmp(argv[1], "compare"))
	{
		FaceRecognition* imgReg = new SvmRegcognition(string(argv[4]));

		cout << imgReg->CompareFace(string(argv[2]), string(argv[3])) << endl;

		delete imgReg;
	}
    return 0;
}

