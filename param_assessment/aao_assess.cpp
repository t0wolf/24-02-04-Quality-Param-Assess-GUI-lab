#include "aao_assess.h"

AAOAssess::AAOAssess(std::string& sEngineFilePath)
	: m_segEnginePath(sEngineFilePath)
{
	m_aaoSegInferer = new AAOSegmentInferer(m_segEnginePath);
}

AAOAssess::~AAOAssess()
{
	delete m_aaoSegInferer;
}

int AAOAssess::doSegInference(cv::Mat& src, std::vector<cv::Mat>& vMasks)
{
	m_aaoSegInferer->doInference(src, vMasks);
	return 1;
}

int AAOAssess::aaoAssessment(cv::Mat& src, std::vector<float>& assessValues)
{
	std::vector<cv::Mat> vMasks;
	doSegInference(src, vMasks);
	postProcess(src, vMasks);

	return 1;
}

int AAOAssess::postProcess(cv::Mat& src, std::vector<cv::Mat>& vMasks)
{
	std::vector<std::vector<cv::Point>> maskContours;
	cv::Mat aaoMask = vMasks[0];
	cv::findContours(aaoMask, maskContours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

	ImageProcess::filterSmallContour(maskContours);
	std::vector<cv::Point> maskContour = maskContours[0];
	cv::RotatedRect maskContourRect = cv::minAreaRect(maskContour);

	// 获得轮廓的最小外接矩形并重新排列矩形各顶点：左上角-右上角-右下角-左下角
	cv::Point2f rectPoints[4];
	maskContourRect.points(rectPoints);
	cv::Point2f orderedPoints[4];
	ImageProcess::orderRotatedRectPoints(rectPoints, orderedPoints);

	cv::Point2f leftMidPoint, rightMidPoint;
	leftMidPoint = (orderedPoints[0] + orderedPoints[3]) / 2.0f;
	rightMidPoint = (orderedPoints[1] + orderedPoints[2]) / 2.0f;
	float midLineSlope = (rightMidPoint.y - leftMidPoint.y) / (rightMidPoint.x - leftMidPoint.x);
	float midLineBias = leftMidPoint.y - leftMidPoint.x * midLineSlope;
	std::vector<float> midLineX = ParamsAssessUtils::linspace(leftMidPoint.x, rightMidPoint.x, 30);
	std::vector<cv::Point2f> midLinePoints;
	for (float x : midLineX)
	{
		float y = x * midLineSlope + midLineBias;
		midLinePoints.emplace_back(x, y);
	}

	float orthSlope = -1 / midLineSlope;
	std::vector<std::pair<cv::Point, cv::Point>> vEndPoints;

	for (auto& midLinePoint : midLinePoints)
	{
		float orthBias = midLinePoint.y - orthSlope * midLinePoint.x;
		int endPointX1, endPointX2;

		endPointX1 = static_cast<int>((midLinePoint.y - 50.f - orthBias) / orthSlope);
		endPointX2 = static_cast<int>((midLinePoint.y + 50.f - orthBias) / orthSlope);
		std::pair<cv::Point, cv::Point> orthLinePoints;
		orthLinePoints.first = cv::Point(endPointX1, static_cast<int>(midLinePoint.y - 50.f));
		orthLinePoints.second = cv::Point(endPointX2, static_cast<int>(midLinePoint.y + 50.f));

		vEndPoints.push_back(orthLinePoints);
	}

	std::vector<std::vector<cv::Point>> interceptPoints;
	ImageProcess::getContourIntersectionPoint(aaoMask, maskContours, vEndPoints, interceptPoints);

	std::pair<int, float> maxDistIdx;
	if (!interceptPoints.empty())
	{
		std::vector<float> vAAODists;
		ParamsAssessUtils::removeAbnormalInterPoints(interceptPoints);
		maxDistIdx = getAAODist(interceptPoints);

		cv::Mat drawLine;
		cv::cvtColor(aaoMask, drawLine, cv::COLOR_GRAY2BGR);
		cv::line(drawLine, interceptPoints[maxDistIdx.first][0], interceptPoints[maxDistIdx.first][1], cv::Scalar(0, 255, 0), 2);
		//cv::imshow("test", drawLine);
		//cv::waitKey(0);
	}
	else
	{
		maxDistIdx = std::make_pair(0, 0.0f);
	}

	return 1;
}

std::pair<int, float> AAOAssess::getAAODist(std::vector<std::vector<cv::Point>>& vInterPoints)
{
	int midIdx = vInterPoints.size() / 2;
	auto it = vInterPoints.begin() + midIdx - 1;
	float fMaxDist = 0.0f;
	int maxIdx = 0;

	while (it != vInterPoints.end())
	{
		float fCurrDist = ParamsAssessUtils::calcLineDist(*it);
		if (fCurrDist > fMaxDist)
		{
			fMaxDist = fCurrDist;
			maxIdx = std::distance(vInterPoints.begin(), it);
		}
		++it;
	}
	return std::make_pair(maxIdx, fMaxDist);
}
