#pragma once

class Filter
{
public:
	Filter() {}
	virtual ~Filter() {}
	virtual bool Exec(cv::Mat& frame) {
		return true;
	}
};
