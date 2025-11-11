#include "asd_and_sjd_access.h"

ASDAndSJDAssess::ASDAndSJDAssess(std::string& sEngineFilePath) 
	: m_segEnginePath(sEngineFilePath)
{
	m_aaoSegInferer = new AAOSegmentInferer(m_segEnginePath);
}

ASDAndSJDAssess::~ASDAndSJDAssess()
{
	delete m_aaoSegInferer;
}

int ASDAndSJDAssess::doSegInference(cv::Mat& src, std::vector<cv::Mat>& vMasks)
{
	m_aaoSegInferer->doInference(src, vMasks);
	return 1;
}

int ASDAndSJDAssess::asdAndsjdAssessment(cv::Mat& src, std::map<std::string, std::vector<float>>& values, std::map<std::string, cv::Mat>& resultPics)
{
	std::vector<cv::Mat> vMasks;
	doSegInference(src, vMasks);
	if (!vMasks.empty())
		postProcess(src, vMasks, values, resultPics);

	return 1;
}

int ASDAndSJDAssess::postProcess(cv::Mat& src, std::vector<cv::Mat>& vMasks, std::map<std::string, std::vector<float>>& values, std::map<std::string, cv::Mat>& resultPics)
{
	std::vector<std::vector<cv::Point>> maskContours;
	cv::Mat asdAndsjdMask = vMasks[0];
	//cv::imshow("asd sjd mask", asdAndsjdMask);
	//cv::waitKey(0);
	cv::findContours(asdAndsjdMask, maskContours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

	ImageProcess::filterSmallContour(maskContours);
	std::vector<cv::Point> maskContour;
	if (!maskContours.empty())
		maskContour = maskContours[0];
	else
		return 0;

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

	//将轮廓依照中轴线分为上下两部分
	float ParameterA, ParameterB, ParameterC, ParameterD, ParameterNorm;
	ParameterA = rightMidPoint.y - leftMidPoint.y;
	ParameterB = leftMidPoint.x - rightMidPoint.x;
	ParameterC = rightMidPoint.x * leftMidPoint.y - leftMidPoint.x * rightMidPoint.y;
	ParameterNorm = std::sqrt(std::pow(ParameterA, 2) + std::pow(ParameterB, 2));

	std::vector<cv::Point> UpperContourPoints, LowerContourPoints, OnContourPoints;

	for (cv::Point contourPoint : maskContour) 
	{
		ParameterD = ParameterA * contourPoint.x + ParameterB * contourPoint.y + ParameterC;
		if (ParameterD > 0.0f) 
		{
			UpperContourPoints.push_back(contourPoint);
		}
		else if (ParameterD < 0.0f)
		{
			LowerContourPoints.push_back(contourPoint);
		}
		else 
		{
			OnContourPoints.push_back(contourPoint);
		}
	}

	////计算上窦测量点////

	std::reverse(UpperContourPoints.begin(), UpperContourPoints.end());
	std::vector<double> UpperDistances;
	std::vector<cv::Point> SelectedUpperContourPoints;
	cv::Point LeftBottomPoint;

	for (cv::Point upperPoint : UpperContourPoints) 
	{
		double UpperDistance = std::abs(ParameterA * upperPoint.x + ParameterB * upperPoint.y + ParameterC) / ParameterNorm;
		UpperDistances.push_back(UpperDistance);
		if (UpperDistance < 10.)   // 找出距离小于10的轮廓点
		{
			SelectedUpperContourPoints.push_back(upperPoint);
		}
	}
	//auto MinDist = std::min_element(UpperDistances.begin(), UpperDistances.end());  //auto返回的也是值吗还是迭代器
	//int MinDistIdx = std::distance(UpperDistances.begin(), MinDist);

	if (SelectedUpperContourPoints.empty())
		return 0;

	LeftBottomPoint = SelectedUpperContourPoints[0];
	for (cv::Point selectedPoint : SelectedUpperContourPoints)   // 找到上述轮廓点中的左下角点
	{
		if (selectedPoint.y > LeftBottomPoint.y || (selectedPoint.y == LeftBottomPoint.y && selectedPoint.x < LeftBottomPoint.x)) 
		{
			LeftBottomPoint = selectedPoint;
		}
	}

	//将轮廓点和距离以左下角点为起始点重新排列
	auto LeftBottomPointIt = std::find(UpperContourPoints.begin(), UpperContourPoints.end(), LeftBottomPoint);
	int LeftBottomPointIdx = std::distance(UpperContourPoints.begin(), LeftBottomPointIt);
	std::rotate(UpperContourPoints.begin(), LeftBottomPointIt, UpperContourPoints.end());
	std::rotate(UpperDistances.begin(), UpperDistances.begin() + LeftBottomPointIdx, UpperDistances.end());

	//平滑上轮廓距离
	std::vector<double> smoothedUpperDistances;
	ImageProcess::windowAverageScale(UpperDistances, smoothedUpperDistances, 9);

	//寻找上轮廓距离极大值
	//std::vector<double> UpperPeakDistances = Utils::findLocalMaximum(smoothedUpperDistances);  // 根据返回的极值寻找索引的时候，如果该极值出现多次，则容易引起错误
    std::vector<std::pair<double,size_t>> UpperPeakDistancesPair = ParamsAssessUtils::findLocalMaximumPair(smoothedUpperDistances);

	//从现有平稳点中挑出极小值点或删去极大值点
    std::vector<float> GradientUpperDistances = ParamsAssessUtils::gradientOneDimension(smoothedUpperDistances);
    std::vector<size_t> SteadyUpperPointsIdx = ParamsAssessUtils::findIndices(GradientUpperDistances, 0.11f);  // 如果没有找到小于阈值的点会报错
	std::vector<size_t> copySteadyUpperPointsIdx(SteadyUpperPointsIdx);

	for (size_t index: SteadyUpperPointsIdx) 
	//for (int i = 0; i < SteadyUpperPointsIdx.size(); i++) 
	{	
		//size_t index = SteadyUpperPointsIdx[i];
		size_t Interval;
		double CurrentDistance = smoothedUpperDistances[index];
		if (index + 20 > smoothedUpperDistances.size() - 1 || static_cast<int>(index) - 20 < 0) 
		{
			Interval = std::min(smoothedUpperDistances.size() - 1 - index, index);
		}
		else 
		{
			Interval = 20;
		}


		if (CurrentDistance > smoothedUpperDistances[index + Interval] && CurrentDistance > smoothedUpperDistances[index - Interval])
		{
			copySteadyUpperPointsIdx.erase(std::remove(copySteadyUpperPointsIdx.begin(), copySteadyUpperPointsIdx.end(), index), copySteadyUpperPointsIdx.end());
		}
	}
	if (copySteadyUpperPointsIdx.size() == 0) // 那就找极小值
	{
		std::vector<std::pair<double, size_t>> SteadyUpperDistancesPair;
		std::vector<double> SteadyUpperDistances;
		std::vector<size_t> LocalMinimumDistancesIdx;
		for (size_t index : SteadyUpperPointsIdx) 
		{
			SteadyUpperDistancesPair.push_back(std::make_pair(smoothedUpperDistances[index], index));
			SteadyUpperDistances.push_back(smoothedUpperDistances[index]);
		}
		if (SteadyUpperDistancesPair.empty())
			return 0;

        LocalMinimumDistancesIdx = ParamsAssessUtils::findLocalMinimumIdx(SteadyUpperDistancesPair);  // 如果该distance多次重复则不正确，要带着索引一起处理!!!!!!
		if (LocalMinimumDistancesIdx.size() == 0)   // 如果也没有极小值
		{
			auto MinimumDistanceIt = std::min_element(SteadyUpperDistances.begin(), SteadyUpperDistances.end());
			copySteadyUpperPointsIdx.push_back(SteadyUpperDistancesPair[std::distance(SteadyUpperDistances.begin(), MinimumDistanceIt)].second);
		}
		else // 否则将极小值的索引保存
		{
			for (auto& distanceIdx : LocalMinimumDistancesIdx) 
			{
				copySteadyUpperPointsIdx.push_back(distanceIdx);
			}
		}
	}

	// 确定窦部上点
	//float SinusUpperDistance = UpperPeakDistancesPair[0];
	//auto SinusUpperIt = std::find(smoothedUpperDistances.begin(), smoothedUpperDistances.end(), SinusUpperDistance);
	//size_t SinusUpperIdx = std::distance(smoothedUpperDistances.begin(), SinusUpperIt);
	if (UpperPeakDistancesPair.empty())
		return 0;

	size_t SinusUpperIdx = UpperPeakDistancesPair[0].second;

	if (copySteadyUpperPointsIdx.empty())
		return 0;

	// 确定窦管交界上点
	size_t JunctionUpperIdx = 0;
	float ThresholdUpper = 0.07f * static_cast<float>(smoothedUpperDistances.size());
	if (copySteadyUpperPointsIdx.size() == 1 || (copySteadyUpperPointsIdx[0] > SinusUpperIdx && std::abs(static_cast<float>(copySteadyUpperPointsIdx[0] - SinusUpperIdx)) >= ThresholdUpper))
	{
		JunctionUpperIdx = copySteadyUpperPointsIdx[0];
	}
	else 
	{
		for (size_t index : copySteadyUpperPointsIdx) 
		{
			if (index > SinusUpperIdx && static_cast<float>(index - SinusUpperIdx) > ThresholdUpper) 
			{
				JunctionUpperIdx = index;
				break;
			}
		}
	}

	////计算下窦测量点////
	std::vector<double> LowerDistances;

	for (cv::Point lowerPoint : LowerContourPoints)
	{
		double LowerDistance = std::abs(ParameterA * lowerPoint.x + ParameterB * lowerPoint.y + ParameterC) / ParameterNorm;
		LowerDistances.push_back(LowerDistance);
	}

	// 寻找下轮廓距离极大值并过滤
	//std::vector<double> LowerPeakDistances = Utils::findLocalMaximum(LowerDistances);
	//std::vector<double> FilteredLowerPeakDistances;
    std::vector<std::pair<double, size_t>> LowerPeakDistancesPair = ParamsAssessUtils::findLocalMaximumPair(LowerDistances);
	std::vector<size_t> FilteredLowerPeakDistancesIdx;

	double MaxLowerDistance = *std::max_element(LowerDistances.begin(), LowerDistances.end());
	for (auto& distanceIdx : LowerPeakDistancesPair) 
	{
		if (distanceIdx.first > 0.3f * MaxLowerDistance) 
		{
			FilteredLowerPeakDistancesIdx.push_back(distanceIdx.second);
		}
	}

	// 确定窦部下点
	//double SinusLowerDistance = FilteredLowerPeakDistances[0];
	//auto SinusLowerIt = std::find(LowerDistances.begin(), LowerDistances.end(), SinusLowerDistance);
	//size_t SinusLowerIdx = std::distance(LowerDistances.begin(), SinusLowerIt);
	if (FilteredLowerPeakDistancesIdx.empty())
		return 0;

	size_t SinusLowerIdx = FilteredLowerPeakDistancesIdx[0];

	// 平滑下轮廓距离
	std::vector<double> smoothedLowerDistances;
	ImageProcess::windowAverageScale(LowerDistances, smoothedLowerDistances, 9);
	
	// 筛选出平稳点
    std::vector<float> GradientLowerDistances = ParamsAssessUtils::gradientOneDimension(smoothedLowerDistances);
    std::vector<size_t> SteadyLowerPointsIdx = ParamsAssessUtils::findIndices(GradientLowerDistances, 0.25f);

	// 确定窦管交界下点
	size_t JunctionLowerIdx = 0;
	float ThresholdLower = 0.0775f * static_cast<float>(smoothedLowerDistances.size());
	if (SteadyLowerPointsIdx.empty())
		return 0;

	if (SteadyLowerPointsIdx.size() == 1 || (SteadyLowerPointsIdx[0] > SinusLowerIdx && std::abs(static_cast<float>(SteadyLowerPointsIdx[0] - SinusLowerIdx)) >= ThresholdLower))
	{
		JunctionLowerIdx = SteadyLowerPointsIdx[0];
	}
	else
	{
		for (size_t index : SteadyLowerPointsIdx)
		{
			if (index > SinusLowerIdx && static_cast<float>(index - SinusLowerIdx) > ThresholdLower)
			{
				JunctionLowerIdx = index;
				break;
			}
		}
	}

	////画出两条直径////
	cv::Mat DrawLine;
	DrawLine = src.clone();
	cv::Size outputSize(asdAndsjdMask.cols, asdAndsjdMask.rows);
	cv::resize(DrawLine, DrawLine, outputSize);

	//cv::cvtColor(asdAndsjdMask, DrawLine, cv::COLOR_GRAY2BGR);
	cv::Mat drawLineSinus = DrawLine.clone();
	cv::Mat drawLineJunction = DrawLine.clone();

	cv::drawContours(DrawLine, maskContours, -1, cv::Scalar(0, 255, 255), 2);

	cv::line(drawLineSinus, UpperContourPoints[SinusUpperIdx], LowerContourPoints[SinusLowerIdx], cv::Scalar(0, 255, 0), 2);  // 窦部直径（绿）
	std::vector<cv::Point> sinusPoints = { UpperContourPoints[SinusUpperIdx], LowerContourPoints[SinusLowerIdx] };
    float sinusDst = ParamsAssessUtils::calcLineDist(sinusPoints);

	cv::line(drawLineJunction, UpperContourPoints[JunctionUpperIdx], LowerContourPoints[JunctionLowerIdx], cv::Scalar(0, 0, 255), 2);  // 窦管交界（红）
	std::vector<cv::Point> junctionPoints = { UpperContourPoints[JunctionUpperIdx], LowerContourPoints[JunctionLowerIdx] };
    float junctionDst = ParamsAssessUtils::calcLineDist(junctionPoints);

	values.insert({ "ASD", std::vector<float>{sinusDst} });
	values.insert({ "SJD", std::vector<float>{junctionDst} });
	resultPics.insert({ "ASD", drawLineSinus });
	resultPics.insert({ "SJD", drawLineJunction });


	//cv::imshow("test", drawLineSinus);
	//cv::imshow("test1", drawLineJunction);
	//cv::waitKey(0);

	return 1;
}
