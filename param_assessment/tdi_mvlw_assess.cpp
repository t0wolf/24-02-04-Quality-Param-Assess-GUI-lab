#include "tdi_mvlw_assess.h"

TDIMVLWAssess::TDIMVLWAssess(std::string& sObjdetectEnginePath, std::string& sKeyptEnginePath)
	:TDIIVSAssess(sObjdetectEnginePath, sKeyptEnginePath)
{

}

int TDIMVLWAssess::tdimvlwAssessment(cv::Mat& src, std::map<std::string, std::vector<float>>& values, std::map<std::string, cv::Mat>& resultPics)
{
	std::vector<Object> objects;
	std::vector<std::vector<cv::Point>> eaPointsList;
	doInference(src, objects, eaPointsList);
	if (objects.size() < 2)  // 20250513更新：objects小于2个不测，必须大于2个
	{
		return 0;
	}
	postProcess(src, objects, eaPointsList, values, resultPics);

	return 1;
}

int TDIMVLWAssess::postProcess(cv::Mat& src, std::vector<Object>& objects, std::vector<std::vector<cv::Point>>& eaPointsList,
	std::map<std::string, std::vector<float>>& values, std::map<std::string, cv::Mat>& resultPics)
{
	
		int yBaseline = 0;  // 检测框的上边沿y均值
		int yCoordS = 0;  // S点的y坐标均值
		int yCoordE = 0;  // E点的y坐标均值
		int yCoordA = 0;  // A点的y坐标均值
		std::vector<int> vCoordsA, vCoordsE, vCoordsS;

		std::vector<std::vector<cv::Point>> originSEAPoints(eaPointsList);
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
			cv::rectangle(drawPoints, currObj.rect, cv::Scalar(0, 255, 0), 2);
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
	//cv::imshow("test", drawPoints);
	//cv::waitKey(0);

	//float scale = 1.0f;
	//std::vector<float> tdimvlw = { abs(yCoordS - yBaseline) * scale, abs(yCoordE - yBaseline) * scale, abs(yCoordA - yBaseline) * scale };
	////float tdimvlwE = abs(yCoordE - yBaseline) * scale;
	////float tdimvlwA = abs(yCoordA - yBaseline) * scale;

	//values.insert({ "侧壁s", std::vector<float>{ tdimvlw[0] } });
	//values.insert({ "侧壁e", std::vector<float>{ tdimvlw[1] } });
	//values.insert({ "侧壁a", std::vector<float>{ tdimvlw[2] } });
	////values.insert({ "TDIMVLWE", tdimvlwE });
	////values.insert({ "TDIMVLWA", tdimvlwA });
	//resultPics.insert({ "侧壁s", drawPoints });
	//resultPics.insert({ "侧壁e", drawPoints });
	//resultPics.insert({ "侧壁a", drawPoints });

	ParamsAssessUtils::generateFinalResult("CBs", vCoordsS, values, drawPoints, resultPics);
	ParamsAssessUtils::generateFinalResult("CBe", vCoordsE, values, drawPoints, resultPics);
	ParamsAssessUtils::generateFinalResult("CBa", vCoordsA, values, drawPoints, resultPics);

	return 1;
}

