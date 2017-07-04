#pragma once

class FaceRecognition
{
public:
	FaceRecognition() {}
	virtual ~FaceRecognition() {}
	virtual std::string Recognize(cv::Mat& mat) { return ""; }
	virtual void defineFace() {};
};
