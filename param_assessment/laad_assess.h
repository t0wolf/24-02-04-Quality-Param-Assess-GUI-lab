#pragma once
#include "aao_segment_inferer.h"
#include "image_process.h"
#include "assess_utils.h"

//Left atrial anteroposterior diameter 左房前后径
class LAADAssess
{
public:
	LAADAssess(std::string& slaEngineFilePath, std::string& savEngineFilePath);

	~LAADAssess();

	int doSegInference(cv::Mat& src, std::vector<cv::Mat>& vlaMasks, std::vector<cv::Mat>& vavMasks);

	int laadAssessment(cv::Mat& src, std::map<std::string, std::vector<float>>& values, std::map<std::string, cv::Mat>& resultPics);

private:
	int postProcess(cv::Mat& src, std::vector<cv::Mat>& vlaMasks, std::vector<cv::Mat>& vavMasks, std::map<std::string, std::vector<float>>& values, std::map<std::string, cv::Mat>& resultPics);

private:
	std::string m_seglaEnginePath;

	std::string m_segavEnginePath;

	AAOSegmentInferer* m_aaolaSegInferer;

	AAOSegmentInferer* m_aaoavSegInferer;
};
