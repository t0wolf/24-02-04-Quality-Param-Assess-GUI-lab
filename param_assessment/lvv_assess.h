#pragma once

#include "lvv_segment_inferer.h"
#include "assess_utils.h"
#include "image_process.h"
#include <numeric>
#define Pi 3.14159265358979323846

class LVVAssess 
{
public:
	LVVAssess(std::string& sEngineFilePath);

	~LVVAssess();

	int doSegInference(std::vector<cv::Mat>& video, std::vector<cv::Mat>& vMasks);

	int lvvAssessment(std::vector<cv::Mat>& video, std::map<std::string, std::vector<float>>& values, std::map<std::string, cv::Mat>& resultPics);

private:
	int postProcess(std::vector<cv::Mat>& video, std::vector<cv::Mat>& vMasks,
		std::map<std::string, std::vector<float>>& values, std::map<std::string, cv::Mat>& resultPics);

private:
	std::string m_segEnginePath;

	LVVSegmentInferer* m_lvvSegInferer;

};