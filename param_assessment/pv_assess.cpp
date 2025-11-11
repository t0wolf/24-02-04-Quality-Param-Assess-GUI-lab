#include "pv_assess.h"

PVAssess::PVAssess(std::string& sObjdetectEnginePath, std::string& sKeyptEnginePath)
	:m_objdetectEnginePath(sObjdetectEnginePath)
	,m_keyptEnginePath(sKeyptEnginePath)
{
	m_pPvObjdetectInferer = new ParamROIDetection(m_objdetectEnginePath);
	m_pPvKeyptInferer = new PVKeypointsInferer(m_keyptEnginePath);
}

PVAssess::~PVAssess()
{
	delete m_pPvObjdetectInferer;
	delete m_pPvKeyptInferer;
}

int PVAssess::doInference(cv::Mat& src, std::vector<Object>& objects, std::vector<std::vector<cv::Point>>& eaPointsList)
{
	std::vector<cv::Mat> croppedImgs;
	objects = m_pPvObjdetectInferer->doInference(src);
	cropObjects(src, objects, croppedImgs);
	for (auto& croppedImg : croppedImgs)
	{
		//cv::imshow("crop", croppedImg);
		//cv::waitKey(0);

		std::vector<cv::Point> eaPoints;
		m_pPvKeyptInferer->doInference(croppedImg, eaPoints);
		eaPointsList.push_back(eaPoints);
	}

	return 0;
}

int PVAssess::pvAssessment(cv::Mat& src, std::map<std::string, std::vector<float>>& values, std::map<std::string, cv::Mat>& resultPics)
{
	std::vector<Object> objects;
	std::vector<std::vector<cv::Point>> eaPointsList;
	doInference(src, objects, eaPointsList);
	if (objects.size() < 2)
	{
		return 0;
	}
	postProcess(src, objects, eaPointsList, values, resultPics);

	return 1;
}

int PVAssess::cropObjects(cv::Mat& src, std::vector<Object>& objects, std::vector<cv::Mat>& croppedImgs)
{
	cv::Mat cropImg = src.clone();
	for (auto& object : objects)
	{
		cv::Mat croppedImg = cropImg(object.rect);
		croppedImgs.push_back(croppedImg);
	}
	return 1;
}

int PVAssess::postProcess(cv::Mat& src, std::vector<Object>& objects, std::vector<std::vector<cv::Point>>& eaPointsList, std::map<std::string,
	std::vector<float>>& values, std::map<std::string, cv::Mat>& resultPics)
{
	int yBaseline = 0;  // 检测框的上边沿y均值
	int yCoordPv = 0;  // pv点的y坐标均值
	std::vector<int> vCoordPV;
	//int yCoordA = 0;  // A点的y坐标均值
	std::vector<std::vector<cv::Point>> originEAPoints(eaPointsList);
	for (int i = 0; i < objects.size(); i++) // 计算检测到的点在原图上的坐标
	{
		int currBaseline = objects[i].rect.y;
		yBaseline += currBaseline;

		originEAPoints[i][0].x += objects[i].rect.x;  // 0为pv
		originEAPoints[i][0].y += objects[i].rect.y;
		yCoordPv += originEAPoints[i][0].y;
		vCoordPV.push_back(originEAPoints[i][0].y - currBaseline);

		//originEAPoints[i][1].x += objects[i].rect.x;  // 1为A
		//originEAPoints[i][1].y += objects[i].rect.y;
		//yCoordA += originEAPoints[i][1].y;
	}
	yBaseline /= objects.size();
	yCoordPv /= objects.size();
	//yCoordA /= objects.size();

	cv::Mat drawPoints = src.clone();
	for (auto& points : originEAPoints)
	{
		// E点
		cv::circle(drawPoints, points[0], 4, cv::Scalar(0, 255, 0), -1);
		cv::putText(drawPoints, "PV", cv::Point(points[0].x, points[0].y - 20), cv::FONT_HERSHEY_COMPLEX, 1, cv::Scalar(0, 255, 0), 2);
		// A点
		//cv::circle(drawPoints, points[1], 4, cv::Scalar(0, 0, 255), -1);
		//cv::putText(drawPoints, "A", cv::Point(points[1].x, points[1].y - 20), cv::FONT_HERSHEY_COMPLEX, 1, cv::Scalar(0, 0, 255), 2);
	}
	//cv::imshow("test", drawPoints);
	//cv::waitKey(0);

	//float scale = 1.0f;
	//int nOutlier = 0;
	//int pointsSize = vCoordPV.size();
	//vCoordPV = ParamsAssessUtils::filterOutliers(vCoordPV, 1.5, nOutlier);
	//int currThresh = static_cast<int>(std::floor(static_cast<float>(pointsSize) * 0.4f));
	//if (nOutlier >= currThresh)
	//	return 0;

	////yCoordPv = *std::max_element(vCoordPV.begin(), vCoordPV.end());
	//yCoordPv = ParamsAssessUtils::findMedium(vCoordPV);
	//std::vector<float> mvPV = { (yCoordPv - yBaseline) * scale };
	////float mvA = (yBaseline - yCoordA) * scale;

	//values.insert({ "PA", mvPV });
	////values.insert({ "MVA", mvA });
	//resultPics.insert({ "PA", drawPoints });

	ParamsAssessUtils::generateFinalResult("PA", vCoordPV, values, drawPoints, resultPics);

	return 1;
}



