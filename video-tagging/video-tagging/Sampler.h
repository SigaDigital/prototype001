#pragma once

class Sampler
{
public:
	Sampler(char* path);
	virtual bool Sampler::Next(cv::Mat& frame);
private:
	cv::VideoCapture m_cap;
	long m_lMs;
};
