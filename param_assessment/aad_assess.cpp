#include "aad_assess.h"

AADAssess::AADAssess(std::string& sEngineFilePath)
	: m_segEnginePath(sEngineFilePath)
{
	m_aaoSegInferer = new AAOSegmentInferer(m_segEnginePath);
}

AADAssess::~AADAssess()
{
	delete m_aaoSegInferer;
}

int AADAssess::doSegInference(cv::Mat& src, std::vector<cv::Mat>& vMasks)
{
	m_aaoSegInferer->doInference(src, vMasks);
	return 1;
}

int AADAssess::aadAssessment(cv::Mat& src, std::map<std::string, std::vector<float>>& values, std::map<std::string, cv::Mat>& resultPics)
{
	std::vector<cv::Mat> vMasks;
	doSegInference(src, vMasks);
	postProcess(src, vMasks, values, resultPics);

	return 1;
}

int AADAssess::postProcess(cv::Mat& src, std::vector<cv::Mat>& vMasks, std::map<std::string, std::vector<float>>& values, std::map<std::string, cv::Mat>& resultPics)
{
	std::vector<std::vector<cv::Point>> maskContours;
	cv::Mat aadMask = vMasks[0];
	cv::resize(aadMask, aadMask, src.size());
	//cv::imshow("aad mask", aadMask);
	//cv::waitKey(0);
	cv::findContours(aadMask, maskContours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

	ImageProcess::filterSmallContour(maskContours);
	if (maskContours.empty())
		return 0;

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
	float midLineBias = leftMidPoint.y - midLineSlope * leftMidPoint.x;
	std::vector<float> midLineX = ParamsAssessUtils::linspace(leftMidPoint.x, rightMidPoint.x, 30);

	//得到中轴线上的点坐标
	std::vector<cv::Point2f> midLinePoints;
	for (float x : midLineX) 
	{
		float y = midLineSlope * x + midLineBias;
		midLinePoints.emplace_back(x, y);
	}

	// 计算得到中轴线上每个点的垂线与最小外接矩形的纵坐标扩大50上下限的交点
	float orthSlope = -1 / midLineSlope;
	std::vector<std::pair<cv::Point, cv::Point>> vEndPoints;

	for (auto& midLinePoint : midLinePoints) 
	{
		float orthBias = midLinePoint.y - orthSlope * midLinePoint.x;
		int endPointX1, endPointX2;

		endPointX1 = static_cast<int>((orderedPoints[1].y - 50.f - orthBias) / orthSlope);
		endPointX2 = static_cast<int>((orderedPoints[3].y + 50.f - orthBias) / orthSlope);
		std::pair<cv::Point, cv::Point> orthLinePoints;
		orthLinePoints.first = cv::Point(endPointX1, static_cast<int>(orderedPoints[1].y - 50.f));
		orthLinePoints.second = cv::Point(endPointX2, static_cast<int>(orderedPoints[3].y + 50.f));

		vEndPoints.push_back(orthLinePoints);
	}

	//找到每条垂线与轮廓的交点/找到所有垂线与轮廓的交点(只保留了与轮廓仅有两个交点的直线的点坐标)
	std::vector<std::vector<cv::Point>> interceptPoints;
	ImageProcess::getContourIntersectionPoint(aadMask, maskContours, vEndPoints, interceptPoints);  // 是否要放到for循环里面，得看一下get函数的实现确定

	std::pair<int, float> maxDistIdx;
	if(!interceptPoints.empty())
	{
		ParamsAssessUtils::removeAbnormalInterPoints(interceptPoints);
		maxDistIdx = getAADDist(interceptPoints);

		cv::Mat drawLine = src.clone();
		cv::Size outputSize(512, 512);
		//cv::cvtColor(aadMask, drawLine, cv::COLOR_GRAY2BGR);
		cv::resize(drawLine, drawLine, outputSize);

		cv::line(drawLine, interceptPoints[maxDistIdx.first][0], interceptPoints[maxDistIdx.first][1], cv::Scalar(0, 255, 0), 2);
		std::vector<cv::Point> aadPoints = { interceptPoints[maxDistIdx.first][0], interceptPoints[maxDistIdx.first][1] };
		float aadDst = ParamsAssessUtils::calcLineDist(aadPoints);

		values.insert({ "AAD", std::vector<float>{aadDst} });
		resultPics.insert({ "AAD" , drawLine });

		//cv::imshow("test", drawLine);
		//cv::waitKey(0);
	}
	else
	{
		maxDistIdx = std::make_pair(0, 0.0f);
	}

	return 1;
}

std::pair<int, float> AADAssess::getAADDist(std::vector<std::vector<cv::Point>>& vInterPoints) 
{
	int midIdx = vInterPoints.size() / 2;

	auto it = vInterPoints.begin();
	if (midIdx)
		it = vInterPoints.begin() + midIdx - 1;
	float fMaxDist = 0.0f;
	int maxIdx = 0;

	//for (auto it = vInterPoints.begin();it != vInterPoints.end();++it)
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