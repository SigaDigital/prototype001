#include "stdafx.h"
#include "sighthound-recognition.h"

SighthoundRecognition::SighthoundRecognition(std::string apiKey)
{
	m_apiKey = std::string("X-Access-Token: ") + apiKey;
	curl_global_init(CURL_GLOBAL_ALL);
}

std::string SighthoundRecognition::Recognize(cv::Mat& mat)
{
	m_label.clear();

	std::string response;
	std::vector<uchar> buffer;
	cv::imencode(".jpg", mat, buffer);
	m_curlHandle = curl_easy_init();
	struct curl_slist* headers = NULL;
	curl_easy_setopt(m_curlHandle, CURLOPT_URL, "https://dev.sighthoundapi.com/v1/recognition?groupId=prototype");
	//curl_easy_setopt(m_curlHandle, CURLOPT_URL, "http://localhost:8888");
	curl_easy_setopt(m_curlHandle, CURLOPT_SSL_VERIFYPEER, 0L);
	curl_easy_setopt(m_curlHandle, CURLOPT_SSL_VERIFYHOST, 0L);
	headers = curl_slist_append(headers, "Content-Type: application/octet-stream");
	headers = curl_slist_append(headers, m_apiKey.c_str());
	curl_easy_setopt(m_curlHandle, CURLOPT_HTTPHEADER, headers);
	curl_easy_setopt(m_curlHandle, CURLOPT_POSTFIELDSIZE, buffer.size());
	curl_easy_setopt(m_curlHandle, CURLOPT_POSTFIELDS, &buffer[0]);
	curl_easy_setopt(m_curlHandle, CURLOPT_WRITEFUNCTION, SighthoundRecognition::WriteCB);
	curl_easy_setopt(m_curlHandle, CURLOPT_WRITEDATA, &response);
	curl_easy_perform(m_curlHandle);
	curl_easy_cleanup(m_curlHandle);

	auto result = nlohmann::json::parse(response.c_str());
	std::string label = result["objects"][0]["objectId"];
	double confidence = result["objects"][0]["faceAnnotation"]["recognitionConfidence"];

	if (confidence > 0.9)
	{
		m_label = label;
	}
	else
	{
		m_label = "";
	}

	return m_label;
}

size_t SighthoundRecognition::WriteCB(void* contents, size_t size, size_t nmemb, void* userp)
{
	std::string* s = (std::string*) userp;
	size_t newLength = size*nmemb;
	size_t oldLength = s->size();
	try
	{
		s->resize(oldLength + newLength);
	}
	catch (std::bad_alloc &e)
	{
		//handle memory problem
		std::cerr << e.what() << std::endl;
		return 0;
	}

	std::copy((char*)contents, (char*)contents + newLength, s->begin() + oldLength);
	return size * nmemb;
}