#pragma once
#include "pv_assess.h"

class TRAssess : public PVAssess
{
public:
	TRAssess(std::string& sObjdetectEnginePath, std::string& sKeyptEnginePath);

	int trAssessment(cv::Mat& src, std::map<std::string, std::vector<float>>& values, std::map<std::string, cv::Mat>& resultPics);

	int postProcess(cv::Mat& src, std::vector<Object>& objects, std::vector<std::vector<cv::Point>>& eaPointsList,
		std::map<std::string, std::vector<float>>& values, std::map<std::string, cv::Mat>& resultPics);
};