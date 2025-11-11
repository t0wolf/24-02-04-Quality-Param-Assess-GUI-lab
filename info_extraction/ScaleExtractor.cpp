#include "ScaleExtractor.h"

ScaleExtractor::ScaleExtractor()
	: m_ocrInferer(new OCRInferer("D:/Resources/20240221/ocr_models/ch_det.engine", 
        "D:/Resources/20240221/ocr_models/ch_rec.engine", "D:/Resources/20240221/en_dict.txt"))
{

}

ScaleExtractor::~ScaleExtractor()
{
	if (m_ocrInferer != nullptr)
	{
		delete m_ocrInferer;
		m_ocrInferer = nullptr;
	}
}

float ScaleExtractor::extractScale(cv::Mat& src)
{
	std::vector<std::string> vTexts;
	std::vector<cv::Rect> vTextROIs;

	extractText(src, vTextROIs, vTexts);
	float scale = getRealScale(vTexts, vTextROIs);
	return scale;
}

int ScaleExtractor::extractText(cv::Mat& src, std::vector<cv::Rect>& vTextROIs, std::vector<std::string>& vTexts)
{
	m_ocrInferer->doInference(src, vTextROIs, vTexts);
	return 1;
}

float ScaleExtractor::getRealScale(std::vector<std::string>& vTexts, std::vector<cv::Rect>& vTextROIs)
{
	if (vTexts.size() != vTextROIs.size())
		return 0;
	std::vector<int> vLengths;
	std::vector<cv::Point> vCtrPoints;
	std::vector<float> vScales;

	for (int i = 0; i < vTexts.size(); i++)
	{
		int currLength = 0;
		try
		{
			currLength = std::stoi(vTexts[i]);
		}
		catch (const std::invalid_argument& ia)
		{
			currLength = -100;
		}

		if (currLength < 0)
			continue;
		
		cv::Rect currRoiRect = vTextROIs[i];
		cv::Point currCtrPoint;
		currCtrPoint.x = currRoiRect.x + currRoiRect.width / 2;
		currCtrPoint.y = currRoiRect.y + currRoiRect.height / 2;

		vLengths.push_back(currLength);
		vCtrPoints.push_back(currCtrPoint);
	}

	for (int i = 1; i < vLengths.size(); i++)
	{
		int currLength = vLengths[i];
		int prevLength = vLengths[i - 1];
		int realDist = std::abs(currLength - prevLength);

		cv::Point currCtrPoint = vCtrPoints[i];
		cv::Point prevCtrPoint = vCtrPoints[i - 1];
		cv::Point2f v = currCtrPoint - prevCtrPoint;
		float fDist = sqrt(v.x * v.x + v.y * v.y);

		float currScale = fDist / static_cast<float>(realDist);
		vScales.push_back(currScale);
	}

	float sum = std::accumulate(vScales.begin(), vScales.end(), 0.0f);
	float scale = sum / vScales.size();

	return scale;
}


