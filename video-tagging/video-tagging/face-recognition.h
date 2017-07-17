#pragma once

class FaceRecognition
{
public:
	FaceRecognition() {}
	virtual ~FaceRecognition() {}
	virtual std::string Recognize(cv::Mat& mat) { return ""; }
	virtual void clustering() {};
	virtual std::string ImgRecognize(std::string img_file_path) { return ""; }
};
