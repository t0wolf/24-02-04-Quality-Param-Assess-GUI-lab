#include "PLAXAorticDetect.h"

PLAXAorticDetect::PLAXAorticDetect(std::string& strEnginePath)
	: ROIDetection(strEnginePath, cv::Size(640, 640))
{
	m_classes = 2;
	m_fConfThresh = 0.2f;
}

std::vector<Object> PLAXAorticDetect::doInference(cv::Mat& src)
{
	std::vector<Object> vecObjects = ROIDetection::doInference(src);
	return vecObjects;
}
