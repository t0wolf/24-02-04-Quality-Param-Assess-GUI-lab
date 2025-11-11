#pragma once
#include "ed_es_line_inferer.h"
#include "ed2value_es_line_inferer.h"
#include "ed4value_es_line_inferer.h"
#include "ed_multi_inferer.h"
#include "image_process.h"
#include "assess_utils.h"

class MultiLineAssess 
{
public:
	MultiLineAssess(std::string& sEnginePath, std::string& sDetachTwoEnginePath, std::string& sDetachFourEnginePath, std::string& sMultiLineMaskPath);

	~MultiLineAssess();

	int doHeatmapInference(std::vector<cv::Mat>& video, std::vector<std::vector<cv::Mat>>& vMasks);

	int doTwoHeatmapInference(std::vector<cv::Mat>& video, std::vector<std::vector<cv::Mat>>& vMasks);

	int doFourHeatmapInference(std::vector<cv::Mat>& video, std::vector<std::vector<cv::Mat>>& vMasks);

	int doMultiLineMaskInference(std::vector<cv::Mat>& video, std::vector<std::vector<cv::Mat>>& vMasks);

	int multiLineAssessment(std::vector<cv::Mat>& video, std::map<std::string, std::vector<float>>& values, std::map<std::string, cv::Mat>& resultPics);

	int detachLineAssessment(std::vector<cv::Mat>& video, std::map<std::string, std::vector<float>>& values, std::map<std::string, cv::Mat>& resultPics);

	int multiLineMaskAssessment(std::vector<cv::Mat>& video, std::map<std::string, std::vector<float>>& values, std::map<std::string, cv::Mat>& resultPics);

	void drawCross(cv::Mat& image, const cv::Point& center, int length, const cv::Scalar& color, int thickness);

	void drawDashedLine(cv::Mat img, cv::Point p1, cv::Point p2, cv::Scalar color, int thickness);

	int setScaleInfo(float& scaleLength, float& scale);

	std::map<std::string, std::vector<float>> getPresentationValues();

	std::map<std::string, cv::Mat> getPresentationPics();
private:
	int postProcess(std::vector<cv::Mat>& video, std::vector<std::vector<cv::Mat>>& vMasks, std::map<std::string, std::vector<float>>& values, std::map<std::string, cv::Mat>& resultPics);

	int postProcessTwoHeatmap(std::vector<cv::Mat>& video, std::vector<std::vector<cv::Mat>>& vMasks, std::map<std::string, std::vector<float>>& values, std::map<std::string, cv::Mat>& resultPics);

	int postProcessFourHeatmap(std::vector<cv::Mat>& video, std::vector<std::vector<cv::Mat>>& vMasks, std::map<std::string, std::vector<float>>& values, std::map<std::string, cv::Mat>& resultPics);

	int postProcessMultiLineMask(std::vector<cv::Mat>& video, std::vector<std::vector<cv::Mat>>& vMasks, std::map<std::string, std::vector<float>>& values, std::map<std::string, cv::Mat>& resultPics);

	void processPoints(cv::Point2f rectPoints[4]);

	int parseFinalResults(std::map<std::string, std::vector<float>>& values, std::map<std::string, std::vector<float>>& realValues, std::map<std::string, cv::Mat>& resultPics);
private:
	float m_scaleLength;
	float m_scale;
	std::string m_heatmapPath;
	std::string m_detachTwoHeatmapPath;
	std::string m_detachFourHeatmapPath;
	std::string m_multiLineMaskPath;
	std::map<std::string, std::vector<float>> m_values;
	std::map<std::string, cv::Mat> m_resultPics;
	std::map<std::string, std::pair<float, float>> m_referRange = {
		{ "IVSTd", {0.5f, 1.5f} },
		{ "LVPWTd", {0.5f, 1.5f} },
		{ "LVDd", {3.5f, 6.5f} },
		{ "LAD", {2.5f, 4.0f} }
	};

	LineInferer* m_heatmapInferer;
	DetachTwoLineInferer* m_detachTwoHeatmapInferer;
	DetachFourLineInferer* m_detachFourHeatmapInferer;
	MultiLineMaskInferer* m_multiLineMaskInferer;

};