#pragma once
//#include <opencv2/ximgproc.hpp>
#include "OutPaintInferer.h"
#include "ed_multi_without_ed_infer.h"
#include "image_process.h"
#include "assess_utils.h"

class MultiLineAssess_Aorta
{
public:
	MultiLineAssess_Aorta(std::string& sEnginePath,
		std::string& sMultiLineMaskPath);

	~MultiLineAssess_Aorta();
	int doAortaMultiLineMaskInference(std::vector<cv::Mat>& video, std::vector<std::vector<cv::Mat>>& vMasks);
	int doAortaOutpaintingInference(std::vector<cv::Mat>& video, std::vector<cv::Mat>& outpaint_video);
	int AortaMultiLineMaskAssessment(std::vector<cv::Mat>& video, std::map<std::string, std::vector<float>>& values, std::map<std::string, cv::Mat>& resultPics);
	int parseFinalResults(std::map<std::string, std::vector<float>>& values, std::map<std::string, std::vector<float>>& realValues, std::map<std::string, cv::Mat>& resultPics);

	float computeAngle(const cv::Point2f& v1, const cv::Point2f& v2);

	//float calculateAngle(const cv::Point2f& v1, const cv::Point2f& v2);




	int setScaleInfo(float& scaleLength, float& scale);

private:

	int postProcessAortaMultiLineMask(std::vector<cv::Mat>& video, std::vector<std::vector<cv::Mat>>& vMasks1, std::map<std::string, std::vector<float>>& values, std::map<std::string, cv::Mat>& resultPics, std::vector<cv::Mat>& rotation_video);

	int postProcessAortaOutpaint(std::vector<cv::Mat>& video, std::vector<std::vector<cv::Mat>>& vMasks, std::map<std::string, std::vector<float>>& values, std::map<std::string, cv::Mat>& resultPics,std::vector<cv::Mat>& rotation_video);

	void processPoints(cv::Point2f rectPoints[4]);
	void drawCross(cv::Mat& image, const cv::Point& center, int length, const cv::Scalar& color, int thickness);
	void drawDashedLine(cv::Mat img, cv::Point p1, cv::Point p2, cv::Scalar color, int thickness);
private:
	float m_scaleLength;
	float m_scale;
	std::string m_outpaintPath;
	std::string m_detachTwoHeatmapPath;
	std::string m_detachFourHeatmapPath;
	std::string m_multiLineMaskPath_Aorta;
	std::map<std::string, std::vector<float>> m_values;
	std::map<std::string, cv::Mat> m_resultPics;
	std::map<std::string, std::pair<float, float>> m_referRange = {
		{ "ASD", {1.5f, 4.0f} },
		{ "SJD", {1.5f, 4.0f} },
		{ "AAD", {2.0f, 5.0f} },
	};
	OutPaintInferer* m_outpaint_Inferer;

	MultiLineMaskInferer_ed* m_multiLineMaskInferer_Aorta;

};