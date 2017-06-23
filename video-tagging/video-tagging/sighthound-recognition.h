#pragma once
#include "face-recognition.h"

class SighthoundRecognition : public FaceRecognition
{
public:
	SighthoundRecognition(std::string apiKey);
	virtual ~SighthoundRecognition() {
		curl_global_cleanup();
	}
	virtual std::string Recognize(cv::Mat& mat);

private:
	static size_t WriteCB(void* contents, size_t size, size_t nmemb, void* userp);
	std::string m_label;
	std::string m_apiKey;
	CURL* m_curlHandle;
};
