#include "tdi_ivs_assess.h"

TDIIVSAssess::TDIIVSAssess(std::string& sObjdetectEnginePath, std::string& sKeyptEnginePath)
	:m_objdetectEnginePath(sObjdetectEnginePath)
	, m_keyptEnginePath(sKeyptEnginePath)
{
	m_pTdiivsObjdetectInferer = new ParamROIDetection(sObjdetectEnginePath);
 	m_pTdiivsKeyptInferer = new TDIIVSKeypointsInferer(sKeyptEnginePath);
}

TDIIVSAssess::~TDIIVSAssess()
{
	delete m_pTdiivsObjdetectInferer;
	delete m_pTdiivsKeyptInferer;
}

int TDIIVSAssess::doInference(cv::Mat& src, std::vector<Object>& objects, std::vector<std::vector<cv::Point>>& eaPointsList)
{
	std::vector<cv::Mat> croppedImgs;
	//cv::imshow("src",src);
	//cv::waitKey(0);
	objects = m_pTdiivsObjdetectInferer->doInference(src); //目标检测的
	cropObjects(src, objects, croppedImgs);
	for (auto& croppedImg : croppedImgs)
	{
		//cv::imshow("crop", croppedImg);
		//cv::waitKey(0);

		std::vector<cv::Point> eaPoints;
		m_pTdiivsKeyptInferer->doInference(croppedImg, eaPoints);
		eaPointsList.push_back(eaPoints);
	}

	return 0;
}

int TDIIVSAssess::tdiivsAssessment(cv::Mat& src, std::map<std::string, std::vector<float>>& values, std::map<std::string, cv::Mat>& resultPics)
{
	std::vector<Object> objects;
	std::vector<std::vector<cv::Point>> eaPointsList;
	//cv::imshow("test" ,src);
	//cv::waitKey(0);
	doInference(src, objects, eaPointsList);
	if (objects.size() < 2) 
	{
		//QtLogger::instance().logMessage("[I] TDIMVIVS: Empty");
		return 0;
	}
	postProcess(src, objects, eaPointsList, values, resultPics);
	//QtLogger::instance().logMessage(QString("[I] TDIMVIVS: %1").arg(values["JGe"][0]));
	return 1;
}

int TDIIVSAssess::cropObjects(cv::Mat& src, std::vector<Object>& objects, std::vector<cv::Mat>& croppedImgs)
{
	cv::Mat cropImg = src.clone();
	for (auto& object : objects)
	{
		cv::Mat croppedImg = cropImg(object.rect);
		croppedImgs.push_back(croppedImg);
	}
	return 1;
}

int TDIIVSAssess::postProcess(cv::Mat& src, std::vector<Object>& objects, std::vector<std::vector<cv::Point>>& eaPointsList, 
	std::map<std::string, std::vector<float>>& values, std::map<std::string, cv::Mat>& resultPics)
{
	int yBaseline = 0;  // 检测框的上边沿y均值
	int yCoordS = 0;  // S点的y坐标均值
	int yCoordE = 0;  // E点的y坐标均值
	int yCoordA = 0;  // A点的y坐标均值
	int APoints_false = 0;  // 与E点重合的A点的数量
	std::vector<int> vCoordsA, vCoordsE, vCoordsS;

	std::vector<std::vector<cv::Point>> originSEAPoints(eaPointsList);
	//for (int i = 0; i < objects.size(); i++) // 计算检测到的点在原图上的坐标
	//{
	//	//int currBaseline = 2 * (objects[i].rect.y + objects[i].rect.height);
	//	//int currBaseline = objects[i].rect.y + objects[i].rect.height / 2;
	//	//yBaseline += currBaseline;

	//	double DistanceEA = std::pow(originSEAPoints[i][2].x - originSEAPoints[i][1].x, 2);
	//	DistanceEA += std::pow(originSEAPoints[i][2].y - originSEAPoints[i][1].y, 2);
	//	DistanceEA = std::sqrt(DistanceEA);

	//	if (DistanceEA > 10)
	//	{
	//		originSEAPoints[i][0].x += objects[i].rect.x;  // 0为S
	//		originSEAPoints[i][0].y += objects[i].rect.y;
	//		yCoordS += originSEAPoints[i][0].y;

	//		originSEAPoints[i][1].x += objects[i].rect.x;  // 1为E
	//		originSEAPoints[i][1].y += objects[i].rect.y;
	//		yCoordE += originSEAPoints[i][1].y;

	//		originSEAPoints[i][2].x += objects[i].rect.x;  // 2为A
	//		originSEAPoints[i][2].y += objects[i].rect.y;
	//		yCoordA += originSEAPoints[i][2].y;

	//		originSEAPoints[i][3].x += objects[i].rect.x;  // 3为Baseline
	//		originSEAPoints[i][3].y += objects[i].rect.y;
	//		originSEAPoints[i][4].x = objects[i].rect.x;  // 3为Baseline
	//		originSEAPoints[i][4].y = objects[i].rect.y;
	//		int currBaseline = originSEAPoints[i][3].y;
	//		yBaseline += originSEAPoints[i][3].y;

	//		vCoordsS.push_back(std::abs(originSEAPoints[i][0].y - currBaseline));
	//		vCoordsE.push_back(std::abs(originSEAPoints[i][1].y - currBaseline));
	//		vCoordsA.push_back(std::abs(originSEAPoints[i][2].y - currBaseline));
	//	}
	//	else
	//	{
	//		originSEAPoints[i][0].x = -1;  // 0为S
	//		originSEAPoints[i][0].y = -1;

	//		originSEAPoints[i][1].x = -1;  // 1为E
	//		originSEAPoints[i][1].y = -1;

	//		originSEAPoints[i][2].x = -1;  // 2为A
	//		originSEAPoints[i][2].y = -1;

	//		originSEAPoints[i][3].x = -1;  // 3为Baseline
	//		originSEAPoints[i][3].y = -1;
	//		APoints_false += 1;
	//	}
	//}
	//if (objects.size() - APoints_false > 0)
	//{
	//	yBaseline /= objects.size() - APoints_false;
	//	yCoordS /= objects.size() - APoints_false;
	//	yCoordE /= objects.size() - APoints_false;
	//	yCoordA /= objects.size() - APoints_false;
	//}

	//cv::Mat drawPoints = src.clone();

	//int counter = 0;
	//for (auto& points : originSEAPoints)
	//{
	//	Object currObj = objects[counter];
	//	cv::rectangle(drawPoints, currObj.rect, cv::Scalar(0, 255, 0), 2);
	//	// S点
	//	if (points[0].x > 0 && points[0].y > 0)
	//	{
	//		cv::circle(drawPoints, points[0], 4, cv::Scalar(0, 0, 255), -1);
	//		cv::putText(drawPoints, "S", cv::Point(points[0].x, points[0].y - 20), cv::FONT_HERSHEY_COMPLEX, 1, cv::Scalar(0, 0, 255), 2);
	//	}
	//	// E点
	//	if (points[1].x > 0 && points[1].y > 0)
	//	{
	//		cv::circle(drawPoints, points[1], 4, cv::Scalar(0, 255, 0), -1);
	//		cv::putText(drawPoints, "E", cv::Point(points[1].x, points[1].y - 20), cv::FONT_HERSHEY_COMPLEX, 1, cv::Scalar(0, 255, 0), 2);
	//	}
	//	// A点
	//	if (points[2].x > 0 && points[2].y > 0)
	//	{
	//		cv::circle(drawPoints, points[2], 4, cv::Scalar(0, 255, 255), -1);
	//		cv::putText(drawPoints, "A", cv::Point(points[2].x, points[2].y - 20), cv::FONT_HERSHEY_COMPLEX, 1, cv::Scalar(0, 255, 255), 2);
	//	}
	//	// baseline
	//	if (points[3].x > 0 && points[3].y > 0)
	//	{
	//		cv::circle(drawPoints, points[3], 4, cv::Scalar(0, 160, 160), -1);
	//		cv::putText(drawPoints, "Base_line", cv::Point(points[3].x, points[3].y - 20), cv::FONT_HERSHEY_COMPLEX, 1, cv::Scalar(0, 160, 160), 2);
	//	}
	//	if (points[4].x > 0 && points[4].y > 0)
	//	{
	//		cv::circle(drawPoints, points[4], 4, cv::Scalar(0, 160, 160), -1);
	//		cv::putText(drawPoints, "start", cv::Point(points[4].x, points[4].y - 20), cv::FONT_HERSHEY_COMPLEX, 1, cv::Scalar(0, 160, 160), 2);
	//	}
	//	++counter;
	//}
	//cv::imshow("test", drawPoints);
	//cv::waitKey(0);
	//buffer_250430
	for (int i = 0; i < objects.size(); i++) // 计算检测到的点在原图上的坐标
	{
		//int currBaseline = 2 * (objects[i].rect.y + objects[i].rect.height);
		//int currBaseline = objects[i].rect.y + objects[i].rect.height / 2;
		//yBaseline += currBaseline;

		originSEAPoints[i][0].x += objects[i].rect.x;  // 0为S
		originSEAPoints[i][0].y += objects[i].rect.y;
		yCoordS += originSEAPoints[i][0].y;

		originSEAPoints[i][1].x += objects[i].rect.x;  // 1为E
		originSEAPoints[i][1].y += objects[i].rect.y;
		yCoordE += originSEAPoints[i][1].y;

		originSEAPoints[i][2].x += objects[i].rect.x;  // 2为A
		originSEAPoints[i][2].y += objects[i].rect.y;
		yCoordA += originSEAPoints[i][2].y;

		originSEAPoints[i][3].x += objects[i].rect.x;  // 3为Baseline
		originSEAPoints[i][3].y += objects[i].rect.y;
		int currBaseline = originSEAPoints[i][3].y;
		yBaseline += originSEAPoints[i][3].y;

		vCoordsS.push_back(std::abs(originSEAPoints[i][0].y - currBaseline));
		vCoordsE.push_back(std::abs(originSEAPoints[i][1].y - currBaseline));
		vCoordsA.push_back(std::abs(originSEAPoints[i][2].y - currBaseline));
	}
	yBaseline /= objects.size();
	yCoordS /= objects.size();
	yCoordE /= objects.size();
	yCoordA /= objects.size();

	cv::Mat drawPoints = src.clone();

	int counter = 0;
	for (auto& points : originSEAPoints)
	{
		Object currObj = objects[counter];
		//cv::rectangle(drawPoints, currObj.rect, cv::Scalar(0, 255, 0), 2);
		// S点
		cv::circle(drawPoints, points[0], 4, cv::Scalar(0, 0, 255), -1);
		cv::putText(drawPoints, "S", cv::Point(points[0].x, points[0].y - 20), cv::FONT_HERSHEY_COMPLEX, 1, cv::Scalar(0, 0, 255), 2);
		// E点
		cv::circle(drawPoints, points[1], 4, cv::Scalar(0, 255, 0), -1);
		cv::putText(drawPoints, "E", cv::Point(points[1].x, points[1].y - 20), cv::FONT_HERSHEY_COMPLEX, 1, cv::Scalar(0, 255, 0), 2);
		// A点
		cv::circle(drawPoints, points[2], 4, cv::Scalar(0, 255, 255), -1);
		cv::putText(drawPoints, "A", cv::Point(points[2].x, points[2].y - 20), cv::FONT_HERSHEY_COMPLEX, 1, cv::Scalar(0, 255, 255), 2);

		cv::circle(drawPoints, points[3], 4, cv::Scalar(0, 160, 160), -1);
		cv::putText(drawPoints, "Base_line", cv::Point(points[3].x, points[3].y - 20), cv::FONT_HERSHEY_COMPLEX, 1, cv::Scalar(0, 160, 160), 2);
		++counter;
	}
	//buffer_end

	//cv::imshow("test", drawPoints);
	//cv::waitKey(0);

	//float scale = 1.0f;
	//std::vector<float> tdiivs = { abs(yCoordS - yBaseline) * scale, abs(yCoordE - yBaseline) * scale, abs(yCoordA - yBaseline) * scale };
	////float tdiivsE = abs(yCoordE - yBaseline) * scale;
	////float tdiivsA = abs(yCoordA - yBaseline) * scale;

	//values.insert({ "间隔s", std::vector<float>{tdiivs[0]} });
	//values.insert({ "间隔e", std::vector<float>{tdiivs[1]} });
	//values.insert({ "间隔a", std::vector<float>{tdiivs[2]} });
	////values.insert({ "TDIIVSE", tdiivsE });
	////values.insert({ "TDIIVSA", tdiivsA });
	//resultPics.insert({ "间隔s", drawPoints });
	//resultPics.insert({ "间隔e", drawPoints });
	//resultPics.insert({ "间隔a", drawPoints });

	ParamsAssessUtils::generateFinalResult("JGs", vCoordsS, values, drawPoints, resultPics);
	ParamsAssessUtils::generateFinalResult("JGe", vCoordsE, values, drawPoints, resultPics);
	ParamsAssessUtils::generateFinalResult("JGa", vCoordsA, values, drawPoints, resultPics);

	return 1;
}

