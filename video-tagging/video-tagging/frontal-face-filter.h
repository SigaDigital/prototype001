#pragma once
#include "filter.h"

class FrontalFaceFilter : public Filter
{
public:
	FrontalFaceFilter();
	virtual bool Exec(cv::Mat& frame);
private:
	dlib::frontal_face_detector m_detector;
};