// video-tagging.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "sampler.h"
#include "frontal-face-filter.h"
#include "sighthound-recognition.h"

int main(int argc, char** argv)
{
	if (argc < 2)
	{
		std::cout << "Usage: " << argv[0] << " inputPath [dontStopAtFirstFace]" << std::endl;
		return -1;
	}

	Sampler sampler(argv[1]);
	FrontalFaceFilter filter;
	SighthoundRecognition faceRecog("7796KskdhG1nMlLjaTWh155dYsbeZGqJzsHq");
	cv::Mat frame;
	while (sampler.Next(frame)) 
	{
		if (filter.Exec(frame))
		{
#ifdef _DEBUG
			cv::imshow("frame", frame);
			cv::waitKey(1000);
#endif
			std::cout << faceRecog.Recognize(frame);
			if (argc == 2)
			{
				break;
			}
		}
	}

    return 0;
}

