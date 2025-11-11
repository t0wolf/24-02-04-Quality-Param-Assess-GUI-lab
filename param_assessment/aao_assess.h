#pragma once
#include "aao_segment_inferer.h"
#include "image_process.h"
#include "assess_utils.h"


class AAOAssess
{
public:
	AAOAssess(std::string& sEngineFilePath);

	~AAOAssess();

	int doSegInference(cv::Mat& src, std::vector<cv::Mat>& vMasks);

	int aaoAssessment(cv::Mat& src, std::vector<float>& assessValues);

private:
	int postProcess(cv::Mat& src, std::vector<cv::Mat>& vMasks);

	std::pair<int, float> getAAODist(std::vector<std::vector<cv::Point>>& vInterPoints);

private:
	std::string m_segEnginePath;

	AAOSegmentInferer* m_aaoSegInferer;
};

