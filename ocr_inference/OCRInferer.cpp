#include "OCRInferer.h"

OCRInferer::OCRInferer(std::string sDetModelPath, std::string sRecModelPath, std::string sDictionaryPath)
	: m_pDet(new OCRDet(sDetModelPath))
	, m_pRec(new OCRRec(sRecModelPath, sDictionaryPath))
{
	qInfo() << "OCRInferer";
}

OCRInferer::~OCRInferer()
{
	if (m_pDet != nullptr)
	{
		delete m_pDet;
		m_pDet = nullptr;
	}
}

int OCRInferer::doInference(cv::Mat& src, std::vector<cv::Rect>& vTextROIs, std::vector<std::string>& vTexts)
{
	if (src.empty())
		return 0;

	std::vector<cv::Mat> vCroppedRoI;
	m_pDet->doInference(src, vTextROIs);
	if (vTextROIs.empty())
		return 0;

	getCroppedRoI(src, vTextROIs, vCroppedRoI);
	if (vCroppedRoI.empty())
		return 0;

	m_pRec->doInference(vCroppedRoI, vTexts);
	return 1;
}

int OCRInferer::doInference(cv::Mat& src, std::vector<TextInfo>& vTextInfos)
{
	std::vector<cv::Rect> vTextROIs;
	std::vector<std::string> vTexts;
	doInference(src, vTextROIs, vTexts);

	for (int i = 0; i < vTexts.size(); i++)
	{
		TextInfo currTextInfo = { vTexts[i], vTextROIs[i], -100.0f };
		vTextInfos.push_back(currTextInfo);
	}
	return 0;
}

int OCRInferer::getCroppedRoI(cv::Mat& src, std::vector<cv::Rect>& roiRect, std::vector<cv::Mat>& croppedRoI)
{
	for (auto& rect : roiRect)
	{
		cv::Mat dst = src.clone();
		cv::Mat roi;
		dst(rect).copyTo(roi);
		//cv::imshow("roi", roi);
		//cv::waitKey(0);
		//cv::destroyWindow("roi");
		croppedRoI.push_back(roi);
	}
	return 1;
}
