#pragma once
#include <string>
#include <opencv\cv.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/shape_predictor.h>

class FaceRecognition
{
public:
	FaceRecognition() {}
	virtual ~FaceRecognition() {}
	virtual void Clustering() {}
	virtual std::string CompareFace(std::string firstPath, std::string secondParth) { return ""; }
	virtual std::string ImgRecognize(std::string img_file_path) { return ""; }
	virtual std::string Recognize(cv::Mat& mat) { return ""; }	
};
