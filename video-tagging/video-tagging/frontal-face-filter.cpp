#include "stdafx.h"
#include "frontal-face-filter.h"

FrontalFaceFilter::FrontalFaceFilter()
{
	m_detector = dlib::get_frontal_face_detector();
}

bool FrontalFaceFilter::Exec(cv::Mat& frame)
{
	std::vector<dlib::rectangle> dets = m_detector(dlib::cv_image<dlib::bgr_pixel>(frame));
	return dets.size() > 0;
}

