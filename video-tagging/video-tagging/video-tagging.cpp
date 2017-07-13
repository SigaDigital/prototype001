// video-tagging.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "sampler.h"
#include "frontal-face-filter.h"
#include "sighthound-recognition.h"
#include "svm_regcognition.h"
#include "TrainFace.h"

using namespace std;
using namespace dlib;

int main(int argc, char** argv)
{
	if (argc < 3)
	{
		cout << "For sighthound method" << endl;
		cout << "Usage : " << argv[0] << " [video file path] sighthound" << endl;
		cout << "For svm method" << endl;
		cout << "Usage : " << argv[0] << " [video file path] svm [pre-trined directory path]" << endl;
		return -1;
	}
	
	
	if (strcmp(argv[1], "train") == 0)
	{
		TrainFace *train = new TrainFace(argv[2], argv[3], argv[4]);
		train->train();
		delete train;
	}
	else 
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

		pRecognizer->clustering();

		delete pRecognizer;
		cout << "end" << endl;
	}

    return 0;
}

