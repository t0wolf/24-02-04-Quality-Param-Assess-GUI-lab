#include "laad_assess.h"

LAADAssess::LAADAssess(std::string& slaEngineFilePath, std::string& savEngineFilePath)
	: m_seglaEnginePath(slaEngineFilePath)
	, m_segavEnginePath(savEngineFilePath)
{
	m_aaolaSegInferer = new AAOSegmentInferer(m_seglaEnginePath);
	m_aaoavSegInferer = new AAOSegmentInferer(m_segavEnginePath);
}

LAADAssess::~LAADAssess()
{
	delete m_aaolaSegInferer;
	delete m_aaoavSegInferer;
}

int LAADAssess::doSegInference(cv::Mat& src, std::vector<cv::Mat>& vlaMasks, std::vector<cv::Mat>& vavMasks)
{
	m_aaolaSegInferer->doInference(src, vlaMasks);
	m_aaoavSegInferer->doInference(src, vavMasks);
	return 1;
}

int LAADAssess::laadAssessment(cv::Mat& src, std::map<std::string, std::vector<float>>& values, std::map<std::string, cv::Mat>& resultPics)
{
	std::vector<cv::Mat> vlaMasks, vavMasks;
	doSegInference(src, vlaMasks, vavMasks);
	postProcess(src, vlaMasks, vavMasks, values, resultPics);

	return 1;
}

int LAADAssess::postProcess(cv::Mat& src, std::vector<cv::Mat>& vlaMasks, std::vector<cv::Mat>& vavMasks, std::map<std::string, std::vector<float>>& values, std::map<std::string, cv::Mat>& resultPics)
{
	////=======================左心房部分====================////
	std::vector<std::vector<cv::Point>> laMaskContours;
	cv::Mat laMask = vlaMasks[0];
	cv::findContours(laMask, laMaskContours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

	ImageProcess::filterSmallContour(laMaskContours);
	if (laMaskContours.empty())
		return 0;
	std::vector<cv::Point> laMaskContour = laMaskContours[0];
	cv::RotatedRect laMaskContourRect = cv::minAreaRect(laMaskContour);

	// for test//
	//cv::Mat laContour = src.clone();
	//for (auto& itPoints : laMaskContour)
	//{
	//	cv::cvtColor(laMask, drawLine, cv::COLOR_GRAY2BGR);
	//	cv::circle(laContour, itPoints, 1, cv::Scalar(0, 255, 255), -1);
	//}
	//cv::imshow("test", laContour);
	//cv::imshow("mask", laMask);
	//cv::waitKey(0);


	// 获得轮廓的最小外接矩形并重新排列矩形各顶点：左上角-右上角-右下角-左下角
	cv::Point2f laRectPoints[4];
	laMaskContourRect.points(laRectPoints);
	cv::Point2f laOrderedPoints[4];
	ImageProcess::orderRotatedRectPoints(laRectPoints, laOrderedPoints);
	
	cv::Point2f laLeftMidPoint, laRightMidPoint;
	laLeftMidPoint = (laOrderedPoints[0] + laOrderedPoints[3]) / 2.0f;
	laRightMidPoint = (laOrderedPoints[1] + laOrderedPoints[2]) / 2.0f;
	float laMidLineSlope = (laRightMidPoint.y - laLeftMidPoint.y) / (laRightMidPoint.x - laLeftMidPoint.x);
	float laMidLineBias = laLeftMidPoint.y - laMidLineSlope * laLeftMidPoint.x;

	//将轮廓依照中轴线分为上下两部分
	float laParameterA, laParameterB, laParameterC, laParameterD, laParameterNorm;
	laParameterA = laRightMidPoint.y - laLeftMidPoint.y;
	laParameterB = laLeftMidPoint.x - laRightMidPoint.x;
	laParameterC = laRightMidPoint.x * laLeftMidPoint.y - laLeftMidPoint.x * laRightMidPoint.y;
	laParameterNorm = sqrt(pow(laParameterA, 2) + pow(laParameterB, 2));

	std::vector<cv::Point> laUpperContourPoints, laLowerContourPoints, laOnContourPoints;
	for (auto& contourPoint : laMaskContour) 
	{
		laParameterD = laParameterA * contourPoint.x + laParameterB * contourPoint.y + laParameterC;
		if (laParameterD > 0.0f) 
		{
			laUpperContourPoints.push_back(contourPoint);
		}
		else if (laParameterD < 0.0f) 
		{
			laLowerContourPoints.push_back(contourPoint);
		}
		else 
		{
			laOnContourPoints.push_back(contourPoint);
		}
	}

	// 将上轮廓按照x从小到大排序
	//std::sort(upperContourPoints.begin(), upperContourPoints.end(), [](cv::Point& a, cv::Point& b) {
	//	return a.x < b.x;
	//	});
	ImageProcess::pointSortByX(laUpperContourPoints);
	ImageProcess::pointSortByX(laLowerContourPoints);


	//// 计算上轮廓拟合线 ////（拟合操作舍去）
	//源代码逻辑是将上下轮廓线拟合，再在两条拟合直线上采样相同的点数，以每上下两点之间的连线的中点连成的直线视作整个轮廓的中轴线。
	std::vector<cv::Point> midLine;
	std::vector<cv::Point2f> smoothedMidLine;
	std::vector<std::pair<float, float>> vNormals;

	ImageProcess::getLVMidLine(laUpperContourPoints, laLowerContourPoints, midLine);  // 获取中轴线上的点
	ImageProcess::windowAverageCurve(midLine, smoothedMidLine, 9);  // 平滑其上的点
	ImageProcess::calcNormals(smoothedMidLine, vNormals);  // 计算中轴线上的点的法向量

	// for test//
	//cv::Mat drawMidLine = src.clone();
	//for (auto& itPoints : midLine)
	//{

	//	cv::cvtColor(laMask, drawLine, cv::COLOR_GRAY2BGR);
	//	cv::circle(drawMidLine, itPoints, 1, cv::Scalar(0, 255, 255), -1);
	//}
	//cv::imshow("test", drawMidLine);
	//cv::waitKey(0);

	cv::Mat drawImage = src.clone();
	//cv::imshow("test", drawImage);
	//cv::waitKey(0);
	std::vector<std::pair<cv::Point, cv::Point>> endPointList;
	std::vector<std::vector<cv::Point>> laInterPointsList;

	ImageProcess::extendNormalsOnImage(drawImage, smoothedMidLine, vNormals, endPointList);

	ImageProcess::getContourIntersectionPoint(drawImage, laMaskContours, endPointList, laInterPointsList);  // 需要看下细节

	// for test//
	//cv::Mat drawLine(src.size(), CV_8UC1);
	//drawLine.setTo(cv::Scalar(255));
	//cv::Mat drawLine = src.clone();
	//for (auto& itPoints : laInterPointsList) 
	//{
	//	
	//	//cv::cvtColor(laMask, drawLine, cv::COLOR_GRAY2BGR);
	//	cv::line(drawLine, itPoints[0], itPoints[1], cv::Scalar(0, 255, 0), 2);
	//}
	//cv::imshow("test", drawLine);
	//cv::waitKey(0);

	////=====================窦部部分====================////
	std::vector<std::vector<cv::Point>> avMaskContours;
	cv::Mat avMask = vavMasks[0];
	cv::findContours(avMask, avMaskContours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
	if (avMaskContours.empty())
		return 0;

	ImageProcess::filterSmallContour(avMaskContours);
	std::vector<cv::Point> avMaskContour = avMaskContours[0];
	cv::RotatedRect avMaskContourRect = cv::minAreaRect(avMaskContour);

	// 获得轮廓的最小外接矩形并重新排列矩形各顶点：左上角-右上角-右下角-左下角
	cv::Point2f avRectPoints[4];
	avMaskContourRect.points(avRectPoints);
	cv::Point2f avOrderedPoints[4];
	ImageProcess::orderRotatedRectPoints(avRectPoints, avOrderedPoints);

	cv::Point2f avLeftMidPoint, avRightMidPoint;
	avLeftMidPoint = (avOrderedPoints[0] + avOrderedPoints[3]) / 2.0f;
	avRightMidPoint = (avOrderedPoints[1] + avOrderedPoints[2]) / 2.0f;
	float avMidLineSlope = (avRightMidPoint.y - avLeftMidPoint.y) / (avRightMidPoint.x - avLeftMidPoint.x);
	float avMidLineBias = avLeftMidPoint.y - avMidLineSlope * avLeftMidPoint.x;

	//将轮廓依照中轴线分为上下两部分
	float avParameterA, avParameterB, avParameterC, avParameterD, avParameterNorm;
	avParameterA = avRightMidPoint.y - avLeftMidPoint.y;
	avParameterB = avLeftMidPoint.x - avRightMidPoint.x;
	avParameterC = avRightMidPoint.x * avLeftMidPoint.y - avLeftMidPoint.x * avRightMidPoint.y;
	avParameterNorm = sqrt(pow(avParameterA, 2) + pow(avParameterB, 2));

	std::vector<cv::Point> avUpperContourPoints, avLowerContourPoints, avOnContourPoints;
	for (auto& contourPoint : avMaskContour)
	{
		avParameterD = avParameterA * contourPoint.x + avParameterB * contourPoint.y + avParameterC;
		if (avParameterD > 0.0f)
		{
			avUpperContourPoints.push_back(contourPoint);
		}
		else if (avParameterD < 0.0f)
		{
			avLowerContourPoints.push_back(contourPoint);
		}
		else
		{
			avOnContourPoints.push_back(contourPoint);
		}
	}

	if (avLowerContourPoints.empty())
		return 0;

	////计算下窦测量点////
	std::vector<double> avLowerDistances;

	for (cv::Point lowerPoint : avLowerContourPoints)
	{
		double lowerDistance = std::abs(avParameterA * lowerPoint.x + avParameterB * lowerPoint.y + avParameterC) / avParameterNorm;
		avLowerDistances.push_back(lowerDistance);
	}

	// 寻找下轮廓距离极大值并过滤
	std::vector<double> avLowerPeakDistances = ParamsAssessUtils::findLocalMaximum(avLowerDistances);
	std::vector<double> avFilteredLowerPeakDistances;
	float avMaxLowerDistance = *std::max_element(avLowerDistances.begin(), avLowerDistances.end());
	for (auto& distance : avLowerPeakDistances)
	{
		if (distance > 0.3f * avMaxLowerDistance)
		{
			avFilteredLowerPeakDistances.push_back(distance);
		}
	}

	if (avFilteredLowerPeakDistances.empty())
		return 0;

	// 确定窦部下点
	double avLowerDistance = avFilteredLowerPeakDistances[0];
	auto avLowerIt = std::find(avLowerDistances.begin(), avLowerDistances.end(), avLowerDistance);
	size_t avLowerIdx = std::distance(avLowerDistances.begin(), avLowerIt);
	cv::Point avPoint = avLowerContourPoints[avLowerIdx];

	// for test
	//cv::Mat drawCircle;
	//cv::cvtColor(avMask, drawCircle, cv::COLOR_GRAY2BGR);
	//cv::circle(drawCircle, avPoint, 5, cv::Scalar(0, 255, 255), -1);
	//cv::imshow("test", drawCircle);
	//cv::waitKey(0);

	// 找到窦部下点和la之间的距离最小点，借此找到穿过窦部下点的laInterPoints，即穿过窦部下点的中值线的垂线
	std::vector<float> laavDistances;
	for (auto& laInterPoints : laInterPointsList) 
	{
		if (laInterPoints.size() > 1) 
		{
			cv::Point laUpperInterPoint = laInterPoints[0];
			float laavDistance = sqrt(pow(laUpperInterPoint.x - avPoint.x, 2) + pow(laUpperInterPoint.y - avPoint.y, 2));
			laavDistances.push_back(laavDistance);
		}
		else 
		{
			continue;
		}
	}

	if (laInterPointsList.empty())
		return 0;

	auto minDistanceIt = std::min_element(laavDistances.begin(), laavDistances.end());
	size_t minDistanceIdx = std::distance(laavDistances.begin(), minDistanceIt);
	std::vector<cv::Point> laavPoints = laInterPointsList[minDistanceIdx];

	cv::Mat drawLine = src.clone();
	cv::Size outputSize(512, 512);
	cv::resize(drawLine, drawLine, outputSize);

	//cv::circle(drawLine, avPoint, 5, cv::Scalar(0, 255, 255), -1);
	cv::line(drawLine, laavPoints[0], laavPoints[1], cv::Scalar(0, 255, 0), 2);  // 应该是可以在drawImage上画的
	float laadDst = ParamsAssessUtils::calcLineDist(laavPoints);

	values.insert({ "LAD", std::vector<float>{laadDst} });
	resultPics.insert({ "LAD", drawLine });

	//cv::imshow("test", drawLine);
	//cv::waitKey(0);

	return 1;
}