#pragma once
#include "aao_segment_inferer.h"
#include "image_process.h"
#include "assess_utils.h"

//Aortic sinus diameter and sinus junction diameter 主动脉窦部直径和窦部交界处直径
class ASDAndSJDAssess
{
public:
	ASDAndSJDAssess(std::string& sEngineFilePath);

	~ASDAndSJDAssess();

	int doSegInference(cv::Mat& src, std::vector<cv::Mat>& vMasks);

	int asdAndsjdAssessment(cv::Mat& src, std::map<std::string, std::vector<float>>& values, std::map<std::string, cv::Mat>& resultPics);

private:
	int postProcess(cv::Mat& src, std::vector<cv::Mat>& vMasks, std::map<std::string, std::vector<float>>& values, std::map<std::string, cv::Mat>& resultPics);

private:
	std::string m_segEnginePath;

	AAOSegmentInferer* m_aaoSegInferer;
};
