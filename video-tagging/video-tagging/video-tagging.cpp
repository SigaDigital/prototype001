// video-tagging.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "sampler.h"
#include "frontal-face-filter.h"
#include "sighthound-recognition.h"
#include "Descriptor.h"
#include "svm_regcognition.h"
#include "Manage.h"

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
	SighthoundRecognition faceRecog("7796KskdhG1nMlLjaTWh155dYsbeZGqJzsHq");
	while (sampler.Next(frame)) 
	{
		if (filter.Exec(frame))
		{	
			if (!strcmp(argv[2], "sighthound"))
			{			
				std::cout << faceRecog.Recognize(frame);
			}
			else if (!strcmp(argv[2], "svm"))
			{
				Descriptor ex;
				std::vector<matrix<float, 0, 1>> tmp = ex.get_description(frame);		
				static SvmRegcognition face(argv[3]);
				for (int i = 0; i < tmp.size(); i++)
					face.SvmRegcognizer(tmp[i]);
			}		
		}
	}
    return 0;
}

