// video-tagging.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "sampler.h"
#include "frontal-face-filter.h"
#include "sighthound-recognition.h"
#include "svm_regcognition.h"

int main(int argc, char** argv)
{
	if (!(argc == 3 || argc == 4))
	{
		cout << "For sighthound method" << endl;
		cout << "Usage : " << argv[0] << " [video file path] sighthound" << endl;
		cout << "For svm method" << endl;
		cout << "Usage : " << argv[0] << " [video file path] svm [pre-trined directory path]" << endl;
		return -1;
	}

	Sampler sampler(argv[1]);
	FrontalFaceFilter filter;	
	cv::Mat frame;

	FaceRecognition* pRecognizer = 0;
	if (strcmp(argv[2], "sighthound") == 0)
	{
		pRecognizer = new SighthoundRecognition("7796KskdhG1nMlLjaTWh155dYsbeZGqJzsHq");
	}
	else
	{
		pRecognizer = new SvmRegcognition();
	}

	while (sampler.Next(frame)) 
	{
		
		if (filter.Exec(frame))
		{	
			cv::imshow("frame", frame);
			cv::waitKey(1000);
			std::cout << pRecognizer->Recognize(frame) << std::endl;
		}
	}

	pRecognizer->defineFace();
	
	delete pRecognizer;
	cout << "end" << endl;
    return 0;
}

