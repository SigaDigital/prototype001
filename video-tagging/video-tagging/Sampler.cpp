#include "stdafx.h"
#include "Sampler.h"

Sampler::Sampler(char* path) : m_lMs(0)
{
	m_cap.open(path);
}

bool Sampler::Next(cv::Mat& frame)
{
	m_cap.set(cv::CAP_PROP_POS_MSEC, m_lMs);
	m_lMs += 3000;
	return m_cap.read(frame);
}
