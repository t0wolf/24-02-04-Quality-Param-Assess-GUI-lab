#pragma once
//#include "ScaleExtractor.h"
#include "ocr_inference/OCRInferer.h"
#include <regex>
#include "type_define.h"


class InfoExtractor
{
public:
	InfoExtractor();

	~InfoExtractor();

	int doInfoExtract(cv::Mat& src, ScaleInfo& scaleInfo, PatientInfo& patientInfo, ModeInfo& modeInfo);

	int doInfoExtract(cv::Mat& src, RoIScaleInfo& roiScaleInfo, ScaleInfo& scaleInfo, PatientInfo& patientInfo, ModeInfo& modeInfo);

	inline ScaleInfo getScaleInfos()
	{
		ScaleInfo scaleInfo;

		if (m_vScaleInfos.size())
			scaleInfo = { m_vScaleInfos[0].distInfo, m_vScaleInfos[0].unit };
		return scaleInfo;
	}

private:
	int doTextRec(cv::Mat& src);

	int depthDirectRec();

	int specInfoExtract(ScaleInfo& scaleInfo);

	int specScaleRec(int unitPositionIdx, ScaleInfo& scaleInfo, std::vector<std::pair<float, cv::Rect>>& scaleValues);

	int checkScaleResults(int unitPositionIdx, std::vector<std::pair<float, cv::Rect>>& scaleValues);

	int specUnitRec(std::string& unit);

	int parseScaleInfo();

	int doScaleExtract(cv::Mat& src, ScaleInfo& scaleInfo);

	int doScaleExtract(ScaleInfo& scaleInfo);

	int doPatientInfoExtract(PatientInfo& patientInfo);

	int doModeInfoExtract(ModeInfo& modeInfo);

	int colorModeJudge(ModeInfo& modeInfo);

	void equalizeHistogram(const cv::Mat& inputImage, cv::Mat& outputImage) {
		// 转换为灰度图像
		cv::Mat grayImage;
		cv::cvtColor(inputImage, grayImage, cv::COLOR_BGR2GRAY);

		// 直方图均衡化
		cv::equalizeHist(grayImage, outputImage);

		cv::cvtColor(outputImage, outputImage, cv::COLOR_GRAY2BGR);
	}

	void enhanceContrastBrightness(const cv::Mat& inputImage, cv::Mat& outputImage, double alpha, int beta) {
		// Create a new image with the same type and size as the input image
		outputImage = cv::Mat::zeros(inputImage.size(), inputImage.type());

		// Iterate over each pixel and apply the transformation
		for (int y = 0; y < inputImage.rows; ++y) {
			for (int x = 0; x < inputImage.cols; ++x) {
				for (int c = 0; c < inputImage.channels(); ++c) {
					outputImage.at<cv::Vec3b>(y, x)[c] = cv::saturate_cast<uchar>(alpha * inputImage.at<cv::Vec3b>(y, x)[c] + beta);
				}
			}
		}
	}

private:
	OCRInferer* m_ocrInferer;

	std::vector<TextInfo> m_vTotalTextInfos;

	std::vector<TextInfo> m_vScaleInfos;

	std::vector<std::string> m_dopplerKeywords = { "Doppler", "PW", "CW" };

	std::vector<std::string> m_colorKeywords = { "Color", "彩色" };
};

