#pragma once

class Sampler
{
public:
	Sampler(char* path);
	virtual bool Sampler::Next(cv::Mat& frame);
	virtual void setRate(long time);
private:
	cv::VideoCapture m_cap;
	long rate;
	long m_lMs;
};
