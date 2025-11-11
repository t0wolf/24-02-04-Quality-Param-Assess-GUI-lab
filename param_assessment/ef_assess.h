#pragma once

#include "ef_segment_inferer.h"
#include "ef_video_inferer.h"
#include "assess_utils.h"
#include "image_process.h"
#include <numeric>
#define Pi 3.14159265358979323846

class EFAssess 
{
public:
	EFAssess(std::string& sSegEngineFilePath, std::string& sVideoEngineFilePath);

	~EFAssess();

	int doEFInference(std::vector<cv::Mat>& video, std::vector<int>& framePixels, std::vector<cv::Mat>& frameMasks, std::vector<float>& predScores);

	int efAssessment(std::vector<cv::Mat>& video, std::map<std::string, std::vector<float>>& values, std::map<std::string, cv::Mat>& resultPics);

private:
	int postProcess(std::vector<cv::Mat>& video, std::vector<int>& framePixels, std::vector<cv::Mat> frameMasks, std::vector<float>& predScores, std::map<std::string, std::vector<float>>& values, std::map<std::string, cv::Mat>& resultPics);

	int calcLvVolume(std::vector<cv::Mat>& lvMasks, float& averageVolume);

private:
	std::string m_segEnginePath;

	std::string m_videoEnginePath;

	EFSegmentInferer* m_segInferer;

	EFVideoInferer* m_videoInferer;
};