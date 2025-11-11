#pragma once
#include "aao_segment_inferer.h"
#include "image_process.h"
#include "assess_utils.h"

//Ascending aorta diameter 升主动脉直径
class AADAssess
{
public:
	AADAssess(std::string& sEngineFilePath);

	~AADAssess();

	int doSegInference(cv::Mat& src, std::vector<cv::Mat>& vMasks);

	int aadAssessment(cv::Mat& src, std::map<std::string, std::vector<float>>& values, std::map<std::string, cv::Mat>& resultPics);

private:
	int postProcess(cv::Mat& src, std::vector<cv::Mat>& vMasks, std::map<std::string, std::vector<float>>& values, std::map<std::string, cv::Mat>& resultPics);

	std::pair<int, float> getAADDist(std::vector<std::vector<cv::Point>>& vInterPoints);

private:
	std::string m_segEnginePath;

	AAOSegmentInferer* m_aaoSegInferer;
};
