#pragma once
#include "OCRDet.h"
#include "OCRRec.h"
#include "type_define.h"


class OCRInferer
{
public:
	OCRInferer(std::string sDetModelPath, std::string sRecModelPath, std::string sDictionaryPath);

	~OCRInferer();

	int doInference(cv::Mat& src, std::vector<cv::Rect>& vTextROIs, std::vector<std::string>& vTexts);

	int doInference(cv::Mat& src, std::vector<TextInfo>& vTextInfos);

private:
	int getCroppedRoI(cv::Mat& src, std::vector<cv::Rect>& roiRect, std::vector<cv::Mat>& croppedRoI);

private:
	OCRDet* m_pDet;
	OCRRec* m_pRec;

};

