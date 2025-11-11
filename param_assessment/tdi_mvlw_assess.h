#pragma once
#include "tdi_ivs_assess.h"

class TDIMVLWAssess : public TDIIVSAssess 
{
public:
	TDIMVLWAssess(std::string& sObjdetectEnginePath, std::string& sKeyptEnginePath);

	int tdimvlwAssessment(cv::Mat& src, std::map<std::string, std::vector<float>>& values, std::map<std::string, cv::Mat>& resultPics);

	int postProcess(cv::Mat& src, std::vector<Object>& objects, std::vector<std::vector<cv::Point>>& eaPointsList,
		std::map<std::string, std::vector<float>>& values, std::map<std::string, cv::Mat>& resultPics);
};