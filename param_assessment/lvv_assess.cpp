#include "lvv_assess.h"

LVVAssess::LVVAssess(std::string& sEngineFilePath)
	:m_segEnginePath(sEngineFilePath)
{
	m_lvvSegInferer = new LVVSegmentInferer(m_segEnginePath);
}

LVVAssess::~LVVAssess()
{
	delete m_lvvSegInferer;
}

int LVVAssess::doSegInference(std::vector<cv::Mat>& video, std::vector<cv::Mat>& vMasks)
{
	m_lvvSegInferer->doInference(video, vMasks);
	return 1;
}

int LVVAssess::lvvAssessment(std::vector<cv::Mat>& video, std::map<std::string, std::vector<float>>& values, std::map<std::string, cv::Mat>& resultPics)
{
	std::vector<cv::Mat> vMasks;
	doSegInference(video, vMasks);
	postProcess(video, vMasks, values, resultPics);
	return 1;
}

int LVVAssess::postProcess(std::vector<cv::Mat>& video, std::vector<cv::Mat>& vMasks,
	std::map<std::string, std::vector<float>>& values, std::map<std::string, cv::Mat>& resultPics)
{
	cv::Mat lvvMask = vMasks[0];
	std::vector<std::vector<cv::Point>> maskContours;
	cv::findContours(lvvMask, maskContours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

	ImageProcess::filterSmallContour(maskContours);

	if (maskContours.empty())
		return 0;

	std::vector<cv::Point> maskContour = maskContours[0];
	cv::RotatedRect maskContourRect = cv::minAreaRect(maskContour);

	cv::Point2f rectPoints[4];
	maskContourRect.points(rectPoints);


	std::sort(rectPoints, rectPoints + 4, [](const cv::Point2f& a, const cv::Point2f& b) {
		return a.y < b.y;
		});  // 按照纵坐标从小到大排序

	// for test
	cv::Mat drawLine = lvvMask.clone();
	cv::cvtColor(lvvMask, drawLine, cv::COLOR_GRAY2BGR);

	// 找外界矩形上段和下段与LV轮廓的交点的中点
	std::vector<cv::Point> midLineEndPoints, midLinePoints;
	std::vector<std::pair<cv::Point, cv::Point>> vEndPoints = { {rectPoints[0], rectPoints[1]} /*, {rectPoints[2], rectPoints[3]}*/ };


	for (int i = 0; i < vEndPoints.size(); i++)   // 上点用轮廓与mask的交点
	{
		std::pair<cv::Point, cv::Point> pointsPair = vEndPoints[i];
		int pixelNum = 1;

		std::vector<cv::Point> tempPoints, maxPoints;

		cv::LineIterator lineIt(pointsPair.first, pointsPair.second, 8, true);

		for (int i = 0; i < lineIt.count; ++i, ++lineIt)
		{
			cv::Point currPoint = lineIt.pos();
			if (cv::pointPolygonTest(maskContour, currPoint, false) == 0)
			{
				tempPoints.push_back(currPoint);
				//cv::circle(drawLine, currPoint, 1, cv::Scalar(255, 0, 0), -1);
			}
		}

		if (tempPoints.empty())
			return 0;

		cv::Point selectedPoint = tempPoints[tempPoints.size() / 2];
		midLineEndPoints.push_back(selectedPoint);

		tempPoints.clear();

	}
	cv::Point lowerMidPoint((rectPoints[2].x + rectPoints[3].x) / 2, (rectPoints[2].y + rectPoints[3].y) / 2);  // 下轮廓的点用矩形下段中点
	midLineEndPoints.push_back(lowerMidPoint);

	// for test
	//cv::circle(drawLine, midLineEndPoints[0], 3, cv::Scalar(0, 0, 255), -1);
	//cv::circle(drawLine, midLineEndPoints[1], 3, cv::Scalar(0, 0, 255), -1);
	//cv::imshow("test", drawLine);
	//cv::waitKey(0);

	// 遍历中线上的点，获取法向量，及其与轮廓交点;计算L（腔室高度）和ai（腔室半径）的值，这两个值都需要比例尺
	float length = ParamsAssessUtils::calcLineDist(midLineEndPoints);

	cv::LineIterator midLineIt(midLineEndPoints[0], midLineEndPoints[1], 4, true);
	for (int i = 0; i < midLineIt.count; ++i, ++midLineIt)
	{
		cv::Point currPoint = midLineIt.pos();
		midLinePoints.push_back(currPoint);

	}
	std::vector<std::pair<float, float>> vNormals;
	ImageProcess::calcNormals(midLinePoints, vNormals);

	std::vector<std::pair<cv::Point, cv::Point>> endPointList;
	std::vector<std::vector<cv::Point>> lvInterPointsList;

	ImageProcess::extendNormalsOnImage(drawLine, midLinePoints, vNormals, endPointList);
	ImageProcess::getContourIntersectionPoint(drawLine, maskContours, endPointList, lvInterPointsList);  // 得到各个内径的两个端点

	float lvVolume = 0.0f;  // 利用辛普森法计算左室容积（无心尖二腔切面，以心尖四腔内径代替）
	int num = lvInterPointsList.size();
	for (auto& points : lvInterPointsList)
	{
		cv::line(drawLine, points[0], points[1], cv::Scalar(0, 0, 255), 1);
		float lvInterDia = ParamsAssessUtils::calcLineDist(points);
		lvVolume += pow(lvInterDia, 2) * (length / num);
	}

	lvVolume *= Pi / 4;
	std::vector<float> vecLvv = { lvVolume };

	cv::Mat blendImg;  // 融合原图和mask
	drawLine.setTo(cv::Scalar(0, 0, 255), drawLine > 0);
	cv::Mat src = video.back();
	cv::resize(drawLine, drawLine, cv::Size(src.cols, src.rows));
	cv::addWeighted(src, 0.7, drawLine, 0.3, 0.0, blendImg);

	//cv::imshow("test", blendImg);
	//cv::waitKey(0);

	values.insert({ "LVV", vecLvv });
	resultPics.insert({ "LVV", blendImg });

	return 1;
}

