#include "image_process.h"


int ImageProcess::findMaxAreaConnected(cv::Mat& src, cv::Mat& dst)
{
	cv::Mat labels;
	cv::Mat tempSrc;
	if (src.empty())
		return 0;

	src.convertTo(tempSrc, CV_8UC1);
	cv::threshold(tempSrc, tempSrc, 128, 255, cv::THRESH_BINARY);
	int numLabels = cv::connectedComponents(tempSrc, labels, 8, CV_32S);
	numLabels -= 1;

	if (!numLabels)
		return 0;

	std::vector<int> area(numLabels);
	for (int label = 1; label < numLabels; label++)
	{
		cv::Mat mask = labels == label;
		area[label - 1] = cv::countNonZero(mask);
	}

	int maxLabel = 1;
	int maxArea = area[0];

	for (int label = 1; label < numLabels; ++label)
	{
		if (area[label] > maxArea)
		{
			maxArea = area[label];
			maxLabel = label + 1;
		}
	}

	dst = labels == maxLabel;
	dst = dst * 255;

	//cv::imshow("mask", src);
	//cv::imshow("Max connect", dst);
	//cv::waitKey(0);
	return 1;
}

std::vector<cv::Point> ImageProcess::findMaxContour(std::vector<std::vector<cv::Point>>& contours)
{
	float maxContourArea = 0.0f;
	std::vector<cv::Point> maxContour;
	for (auto& contour : contours)
	{
		float currArea = cv::contourArea(contour);
		if (currArea > maxContourArea)
		{
			maxContourArea = currArea;
			maxContour = contour;
		}
	}
	return maxContour;
}

float ImageProcess::findMaxContourArea(std::vector<std::vector<cv::Point>>& contours)
{
	float maxContourArea = 0.0f;
	for (auto& contour : contours)
	{
		float currArea = cv::contourArea(contour);
		if (currArea > maxContourArea)
			maxContourArea = currArea;
	}
	return maxContourArea;
}

int ImageProcess::filterSmallContour(std::vector<std::vector<cv::Point>>& contours)
{
	float maxContourArea = findMaxContourArea(contours);
	std::vector<std::vector<cv::Point>> resultContours;

	for (auto& contour : contours)
	{
		float currArea = cv::contourArea(contour);
		if (currArea >= maxContourArea * 0.2f)
			resultContours.push_back(contour);
	}

	contours = resultContours;
	return 1;
}

int ImageProcess::filterSmallComp(int numComp, cv::Mat& src, cv::Mat& labeledImage, cv::Mat& stats)
{
	int resultNumComp = 0;
	for (int i = 0; i < numComp; i++)
	{
		int area = stats.at<int>(i, cv::CC_STAT_AREA);
		if (area < 100)
		{
			cv::Mat compMask = (labeledImage == i);
			src.setTo(0, compMask);
			labeledImage.setTo(0, compMask);
		}
		else
		{
			resultNumComp++;
		}
	}
	return resultNumComp;
}

int ImageProcess::getSkeletonizeMask(cv::Mat& mask, cv::Mat& skeleton)
{
	cv::Mat binaryMask;
	cv::Mat maskFill = cv::Mat::zeros(cv::Size(mask.cols + 2, mask.rows + 2), CV_8UC1);
	cv::threshold(mask, binaryMask, 0, 255, cv::THRESH_BINARY);

	cv::Mat imFloodFill = binaryMask.clone();
	cv::Mat imFloodFillInv;
	cv::floodFill(imFloodFill, maskFill, cv::Point(0, 0), cv::Scalar(255));

	cv::bitwise_not(imFloodFill, imFloodFillInv);
	binaryMask = binaryMask | imFloodFillInv;

	//cv::Mat distResult;
	//cv::distanceTransform(binaryMask, distResult, cv::DIST_L2, cv::DIST_MASK_PRECISE);
	//std::cout << distResult;
	//threshold(distResult, binaryMask, 10, 255.0, cv::THRESH_BINARY);

	skeletonize(binaryMask, skeleton);
	std::vector<cv::Point> temp;
	for (int i = 0; i < skeleton.rows; i++)
	{
		for (int j = 0; j < skeleton.cols; j++)
		{
			uchar pixelValue = skeleton.at<uchar>(i, j);

			// 如果像素值为255，将其添加到集合中
			if (pixelValue == 255) {
				temp.emplace_back(j, i);
			}
		}
	}

	temp = removeConcavePart(temp);

	//cv::findContours(skeleton, skelContours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

	// 显示轮廓和凸包
	cv::Mat contourImage = cv::Mat::zeros(skeleton.size(), CV_8UC1);
	for (auto& point : temp)
	{
		contourImage.at<uchar>(point) = 255;
	}

	cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
	cv::dilate(contourImage, contourImage, kernel, cv::Point(-1, -1));
	//cv::erode(contourImage, contourImage, kernel, cv::Point(-1, -1));

	cv::Mat labeledSkel;
	cv::Mat stats;
	cv::Mat centroids;

	int numComp = cv::connectedComponentsWithStats(contourImage, labeledSkel, stats, centroids);

	cv::erode(contourImage, contourImage, kernel, cv::Point(-1, -1));

	numComp = filterSmallComp(numComp, contourImage, labeledSkel, stats);
	contourImage.copyTo(skeleton);
	//cv::imshow("test", skeleton);
	//cv::waitKey(0);

	return 1;
}

int ImageProcess::skeletonize(cv::Mat& binaryMask, cv::Mat& skeleton)
{
	//cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(17, 17));
	//cv::erode(binaryMask, binaryMask, kernel);
	skeleton = binaryMask.clone();
	skeleton /= 255;

	cv::Mat prev = cv::Mat::zeros(binaryMask.size(), CV_8UC1);
	cv::Mat diff;

	do {
		skeletonizeIter(skeleton, 0);
		skeletonizeIter(skeleton, 1);
		cv::absdiff(skeleton, prev, diff);
		skeleton.copyTo(prev);

		//cv::imshow("test", skeleton * 255);
		//cv::waitKey(0);
	} while (cv::countNonZero(diff) > 0);

	skeleton *= 255;

	//distTransformFilter(skeleton);


	//cv::Mat prev = cv::Mat::zeros(skeleton.size(), CV_8UC1);
	//cv::Mat diff;

	//do {
	//	// Step 1
	//	cv::Mat marker = cv::Mat::zeros(skeleton.size(), CV_8UC1);

	//	for (int i = 1; i < skeleton.rows - 1; ++i) {
	//		for (int j = 1; j < skeleton.cols - 1; ++j) {
	//			uchar p2 = skeleton.at<uchar>(i - 1, j);
	//			uchar p3 = skeleton.at<uchar>(i - 1, j + 1);
	//			uchar p4 = skeleton.at<uchar>(i, j + 1);
	//			uchar p5 = skeleton.at<uchar>(i + 1, j + 1);
	//			uchar p6 = skeleton.at<uchar>(i + 1, j);
	//			uchar p7 = skeleton.at<uchar>(i + 1, j - 1);
	//			uchar p8 = skeleton.at<uchar>(i, j - 1);
	//			uchar p9 = skeleton.at<uchar>(i - 1, j - 1);

	//			int A = (p2 == 0 && p3 == 1) + (p3 == 0 && p4 == 1) +
	//				(p4 == 0 && p5 == 1) + (p5 == 0 && p6 == 1) +
	//				(p6 == 0 && p7 == 1) + (p7 == 0 && p8 == 1) +
	//				(p8 == 0 && p9 == 1) + (p9 == 0 && p2 == 1);

	//			int B = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9;

	//			int m1 = (p2 * p4 * p6);
	//			int m2 = (p4 * p6 * p8);

	//			if (A == 1 && (B >= 2 && B <= 6) && m1 == 0 && m2 == 0) {
	//				marker.at<uchar>(i, j) = 1;
	//			}
	//		}
	//	}

	//	skeleton &= ~marker;

	//	// Step 2
	//	marker = cv::Mat::zeros(skeleton.size(), CV_8UC1);

	//	for (int i = 1; i < skeleton.rows - 1; ++i) {
	//		for (int j = 1; j < skeleton.cols - 1; ++j) {
	//			uchar p2 = skeleton.at<uchar>(i - 1, j);
	//			uchar p3 = skeleton.at<uchar>(i - 1, j + 1);
	//			uchar p4 = skeleton.at<uchar>(i, j + 1);
	//			uchar p5 = skeleton.at<uchar>(i + 1, j + 1);
	//			uchar p6 = skeleton.at<uchar>(i + 1, j);
	//			uchar p7 = skeleton.at<uchar>(i + 1, j - 1);
	//			uchar p8 = skeleton.at<uchar>(i, j - 1);
	//			uchar p9 = skeleton.at<uchar>(i - 1, j - 1);

	//			int A = (p2 == 0 && p3 == 1) + (p3 == 0 && p4 == 1) +
	//				(p4 == 0 && p5 == 1) + (p5 == 0 && p6 == 1) +
	//				(p6 == 0 && p7 == 1) + (p7 == 0 && p8 == 1) +
	//				(p8 == 0 && p9 == 1) + (p9 == 0 && p2 == 1);

	//			int B = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9;

	//			int m1 = (p2 * p4 * p8);
	//			int m2 = (p2 * p6 * p8);

	//			if (A == 1 && (B >= 2 && B <= 6) && m1 == 0 && m2 == 0) {
	//				marker.at<uchar>(i, j) = 1;
	//			}
	//		}
	//	}

	//	skeleton &= ~marker;
	//	cv::absdiff(skeleton, prev, diff);
	//	skeleton.copyTo(prev);
	//} while (cv::countNonZero(diff) > 0);

	return 1;
}

int ImageProcess::skeletonizeIter(cv::Mat& img, int iter)
{
	CV_Assert(img.channels() == 1);
	CV_Assert(img.depth() != sizeof(uchar));
	CV_Assert(img.rows > 3 && img.cols > 3);

	cv::Mat marker = cv::Mat::zeros(img.size(), CV_8UC1);

	int nRows = img.rows;
	int nCols = img.cols;

	if (img.isContinuous()) {
		nCols *= nRows;
		nRows = 1;
	}

	int x, y;
	uchar* pAbove;
	uchar* pCurr;
	uchar* pBelow;
	uchar* nw, * no, * ne;    // north (pAbove)
	uchar* we, * me, * ea;
	uchar* sw, * so, * se;    // south (pBelow)

	uchar* pDst;

	for (y = 1; y < img.rows - 1; ++y) {
		pAbove = img.ptr<uchar>(y - 1);
		pCurr = img.ptr<uchar>(y);
		pBelow = img.ptr<uchar>(y + 1);
		pDst = marker.ptr<uchar>(y);

		for (x = 1; x < img.cols - 1; ++x) {
			nw = pAbove + x - 1;
			no = pAbove + x;
			ne = pAbove + x + 1;
			we = pCurr + x - 1;
			me = pCurr + x;
			ea = pCurr + x + 1;
			sw = pBelow + x - 1;
			so = pBelow + x;
			se = pBelow + x + 1;

			int A = (*no == 0 && *ne == 1) + (*ne == 0 && *ea == 1) +
				(*ea == 0 && *se == 1) + (*se == 0 && *so == 1) +
				(*so == 0 && *sw == 1) + (*sw == 0 && *we == 1) +
				(*we == 0 && *nw == 1) + (*nw == 0 && *no == 1);
			int B = *no + *ne + *ea + *se + *so + *sw + *we + *nw;

			int m1 = iter == 0 ? (*no * *ea * *so) : (*no * *ea * *we);
			int m2 = iter == 0 ? (*ea * *so * *we) : (*no * *so * *we);

			if (A == 1 && (B >= 2 && B <= 6) && m1 == 0 && m2 == 0) {
				pDst[x] = 1;
			}
		}
	}

	img &= ~marker;
	return 1;
}

int ImageProcess::distTransformFilter(cv::Mat& src)
{
	//cv::dilate(src, src, cv::Mat(), cv::Point(-1, -1), 2);
	cv::Mat distResult;
	cv::distanceTransform(src, distResult, cv::DIST_L2, cv::DIST_MASK_PRECISE);
	//std::cout << distResult;
	threshold(distResult, src, 2, 255.0, cv::THRESH_BINARY_INV);
	return 1;
}

std::vector<cv::Point> ImageProcess::removeConcavePart(std::vector<cv::Point>& skelContours)
{
	std::vector<cv::Point> resultContours;
	std::vector<int> sampledX;

	for (int i = 0; i < skelContours.size(); i++)
	{
		int x = skelContours[i].x;
		int y = skelContours[i].y;
		if (i > 0 && std::find(sampledX.begin(), sampledX.end(), x) != sampledX.end())
		{
			continue;
		}
		int yTotal = 0;
		int counter = 0;
		bool matchFlag = false;

		for (int j = 0; j < skelContours.size(); j++)
		{
			if (skelContours[j].x == x)
			{
				cv::Point temp = skelContours[j];
				yTotal += temp.y;
				++counter;
			}
		}

		if (yTotal)
			resultContours.emplace_back(x, yTotal / counter);

		else
			resultContours.emplace_back(x, y);

		sampledX.push_back(x);

	}
	return resultContours;
}

int ImageProcess::getSkelPoints(cv::Mat& src, std::vector<cv::Point>& vPoints, int value)
{
	for (int i = 0; i < src.rows; i++)
	{
		for (int j = 0; j < src.cols; j++)
		{
			if (src.at<uchar>(i, j) == value)
			{
				vPoints.emplace_back(j, i);
			}
		}
	}

	return 1;
}

int ImageProcess::getSkelPoints(cv::Mat& src, std::vector<cv::Point>& vPoints, bool value)
{
	for (int i = 0; i < src.rows; i++)
	{
		for (int j = 0; j < src.cols; j++)
		{
			if (src.at<uchar>(i, j) == value)
			{
				vPoints.emplace_back(j, i);
			}
		}
	}

	return 1;
}

int ImageProcess::windowAverageCurve(std::vector<cv::Point>& points, std::vector<cv::Point2f>& smoothedPoints, int windowSize)
{
	// 边界检查
	if (points.size() <= windowSize) {
		for (cv::Point point : points)
			smoothedPoints.push_back(point);
		return 0;  // 点的数量小于等于窗口大小，无法进行平滑，直接返回原始曲线
	}

	for (int i = windowSize / 2; i < points.size() - windowSize / 2; ++i) {
		float sumX = 0.0f;
		float sumY = 0.0f;

		// 计算窗口内点的平均值
		for (int j = i - windowSize / 2; j <= i + windowSize / 2; ++j) {
			sumX += points[j].x;
			sumY += points[j].y;
		}

		float averageX = sumX / windowSize;
		float averageY = sumY / windowSize;

		smoothedPoints.push_back(cv::Point2f(averageX, averageY));
	}
	return 1;
}

int ImageProcess::windowAverageScale(std::vector<double>& scale, std::vector<double>& smoothedScale, int windowSize)
{
	// 边界检查
	if (scale.size() <= windowSize) 
	{
		smoothedScale = scale;
		return 0;  // 点的数量小于等于窗口大小，无法进行平滑，直接返回原始曲线
	}

	for (int i = windowSize / 2; i < scale.size() - windowSize / 2; ++i) {
		float sum = 0.0f;

		// 计算窗口内点的平均值
		for (int j = i - windowSize / 2; j <= i + windowSize / 2; ++j) {
			sum += scale[j];
		}

		float average = sum / windowSize;

		smoothedScale.push_back(average);
	}
	return 1;
}

int ImageProcess::getLVMidLine(std::vector<cv::Point>& ivsPoints, std::vector<cv::Point>& pwPoints, std::vector<cv::Point>& midLine)
{
	if (ivsPoints.empty() || pwPoints.empty())
		return 0;

	int numPointSample = std::min(ivsPoints.size(), pwPoints.size());

	for (int i = 0; i < numPointSample; i++)
	{
		cv::Point ivsPoint = ivsPoints[i];
		cv::Point pwPoint = pwPoints[i];

		cv::Point temp;
		temp.x = (ivsPoint.x + pwPoint.x) / 2;
		temp.y = (ivsPoint.y + pwPoint.y) / 2;
		midLine.push_back(temp);
	}

	return 1;
}

int ImageProcess::getLVMidLine(std::vector<cv::Point2f>& ivsPoints, std::vector<cv::Point2f>& pwPoints, std::vector<cv::Point>& midLine)
{
	if (ivsPoints.empty() || pwPoints.empty())
		return 0;

	int numPointSample = std::min(ivsPoints.size(), pwPoints.size());

	for (int i = 0; i < numPointSample; i++)
	{
		cv::Point2f ivsPoint = ivsPoints[i];
		cv::Point2f pwPoint = pwPoints[i];

		cv::Point temp;
		temp.x = (ivsPoint.x + pwPoint.x) / 2;
		temp.y = (ivsPoint.y + pwPoint.y) / 2;
		midLine.push_back(temp);
	}

	return 1;
}

int ImageProcess::calcNormals(std::vector<cv::Point>& midLine, std::vector<std::pair<float, float>>& vNormals)
{
	for (int i = 0; i < midLine.size(); i++)
	{
		int prevPointIdx = i > 0 ? (i - 1) : (midLine.size() - 1);
		cv::Point prevPoint = midLine[prevPointIdx];
		cv::Point currPoint = midLine[i];
		float x = static_cast<float>(currPoint.x) - static_cast<float>(prevPoint.x);
		float y = static_cast<float>(currPoint.y) - static_cast<float>(prevPoint.y);
		float norm = sqrt(x * x + y * y);

		if (norm != 0.0f)
			vNormals.emplace_back(std::make_pair<float, float>(-y / norm, x / norm));
		else
			vNormals.emplace_back(std::make_pair<float, float>(0.0f, 0.0f));
	}
	return 1;
}

int ImageProcess::calcNormals(std::vector<cv::Point2f>& midLine, std::vector<std::pair<float, float>>& vNormals)
{
	for (int i = 0; i < midLine.size(); i++)
	{
		int prevPointIdx = i > 0 ? (i - 1) : (midLine.size() - 1);
		auto prevPoint = midLine[prevPointIdx];
		auto currPoint = midLine[i];
		auto v = currPoint - prevPoint;
		float norm = sqrt(v.x * v.x + v.y * v.y);

		if (norm != 0.0f)
			vNormals.emplace_back(std::make_pair<float, float>(-v.y / norm, v.x / norm));
		else
			vNormals.emplace_back(std::make_pair<float, float>(0.0f, 0.0f));
	}
	return 1;
}

int ImageProcess::extendNormalsOnImage(cv::Mat& src, std::vector<cv::Point>& skelPoints, std::vector<std::pair<float, float>>& vNormals,
	std::vector<std::pair<cv::Point, cv::Point>>& vEndPoints, int length)
{
	if (skelPoints.size() != vNormals.size() && vNormals.size() != 1)
		return 0;

	for (int i = 0; i < skelPoints.size(); i++)
	{
		cv::Point skelPoint = skelPoints[i];
		std::pair<float, float> normal;
		if (vNormals.size() == 1)
			normal = vNormals[0];
		else
			normal = vNormals[i];

		cv::Point endPointPos, endPointNeg;
		endPointPos.x = skelPoint.x + normal.first * length;
		endPointPos.y = skelPoint.y + normal.second * length;

		endPointNeg.x = skelPoint.x - normal.first * length;
		endPointNeg.y = skelPoint.y - normal.second * length;

		endPointPos = clampPoint(endPointPos, src.cols - 1, src.rows - 1);
		endPointNeg = clampPoint(endPointNeg, src.cols - 1, src.rows - 1);
		std::pair<cv::Point, cv::Point> temp{ endPointPos, endPointNeg };

		vEndPoints.push_back(temp);
	}

	return 1;
}

int ImageProcess::extendNormalsOnImage(cv::Mat& src, std::vector<cv::Point2f>& skelPoints, std::vector<std::pair<float, float>>& vNormals,
	std::vector<std::pair<cv::Point, cv::Point>>& vEndPoints, int length)
{
	if (skelPoints.size() != vNormals.size())
		return 0;

	for (int i = 0; i < skelPoints.size(); i++)
	{
		cv::Point skelPoint = skelPoints[i];
		std::pair<float, float> normal = vNormals[i];
		std::pair<cv::Point, cv::Point> endPoint;
		extendNormal(src, skelPoint, normal, endPoint, length);

		//cv::Point endPointPos, endPointNeg;
		//endPointPos.x = static_cast<int>(skelPoint.x + normal.first * length);
		//endPointPos.y = static_cast<int>(skelPoint.y + normal.second * length);

		//endPointNeg.x = static_cast<int>(skelPoint.x - normal.first * length);
		//endPointNeg.y = static_cast<int>(skelPoint.y - normal.second * length);

		//endPointPos = clampPoint(endPointPos, src.cols - 1, src.rows - 1);
		//endPointNeg = clampPoint(endPointNeg, src.cols - 1, src.rows - 1);
		//std::pair<cv::Point, cv::Point> temp{ endPointPos, endPointNeg };

		vEndPoints.push_back(endPoint);
	}

	return 1;
}

bool ImageProcess::calculateIntersection(float x0, float y0, float dx, float dy, float& t, int imgWidth, int imgHeight)
{
	// 初始化 t 的范围（tmin, tmax）
	float tmin = -std::numeric_limits<float>::infinity();
	float tmax = std::numeric_limits<float>::infinity();

	// 每个边界的判定
	if (dx != 0) { // 垂直边界（x=0 或 x=cols-1）
		float t1 = (0 - x0) / dx;
		float t2 = (imgWidth - 1 - x0) / dx;
		tmin = std::max(tmin, std::min(t1, t2));
		tmax = std::min(tmax, std::max(t1, t2));
	}
	if (dy != 0) { // 水平边界（y=0 或 y=rows-1）
		float t1 = (0 - y0) / dy;
		float t2 = (imgHeight - 1 - y0) / dy;
		tmin = std::max(tmin, std::min(t1, t2));
		tmax = std::min(tmax, std::max(t1, t2));
	}

	// 如果有效范围为空，说明直线与图像没有交点
	if (tmin > tmax) {
		return false;
	}

	// 返回 t 的有效范围
	t = tmax; // tmin 为较近的交点，tmax 为较远的交点
	return true;
}

int ImageProcess::extendNormal(cv::Mat& src, cv::Point& point, std::pair<float, float>& pairNormal, std::pair<cv::Point, cv::Point>& pairEndPoint, int length)
{
	// 原点和方向向量
	float x0 = point.x, y0 = point.y;
	float dx = pairNormal.first, dy = pairNormal.second;

	// 获取图像大小
	int imgWidth = src.cols;
	int imgHeight = src.rows;

	// 正方向：寻找 tmax
	float t;
	if (!calculateIntersection(x0, y0, dx, dy, t, imgWidth, imgHeight)) {
		return 0; // 无交点
	}
	cv::Point endPos(static_cast<int>(x0 + t * dx), static_cast<int>(y0 + t * dy));

	// 反方向：寻找 tmin
	if (!calculateIntersection(x0, y0, -dx, -dy, t, imgWidth, imgHeight)) {
		return 0; // 无交点
	}
	cv::Point endNeg(static_cast<int>(x0 - t * dx), static_cast<int>(y0 - t * dy));

	// 输出结果
	pairEndPoint = std::make_pair(endPos, endNeg);

	return 1;
}

int ImageProcess::getLineRectIntersectionPoint(cv::Mat& src, cv::Rect& boundBox, std::pair<cv::Point, cv::Point>& endPoint, std::vector<cv::Point>& interPoints)
{
	cv::Point p1 = endPoint.first;
	cv::Point p2 = endPoint.second;
	//cv::Point rectInterPoint1, rectInterPoint2;

	//if (p1.x != p2.x)
	//{
	//	float k = static_cast<float>(p2.y - p1.y) / static_cast<float>(p2.x - p1.x);
	//	float b = static_cast<float>(p1.y - p1.x * k);

	//	rectInterPoint1.x = boundBox.x;
	//	rectInterPoint1.y = static_cast<int>(k * static_cast<float>(boundBox.x) * k + b);

	//	rectInterPoint2.x = boundBox.x + boundBox.width;
	//	rectInterPoint2.y = static_cast<int>(k * static_cast<float>(rectInterPoint2.x) + b);
	//}
	//else
	//{
	//	rectInterPoint1.x = p1.x;
	//	rectInterPoint1.y = boundBox.y;

	//	rectInterPoint1.x = p1.x;
	//	rectInterPoint1.y = boundBox.y + boundBox.height;
	//}

	cv::Point rectPoints[4];
	rectPoints[0] = boundBox.tl();                     // 左上角
	rectPoints[1] = cv::Point(boundBox.x + boundBox.width, boundBox.y);  // 右上角
	rectPoints[2] = boundBox.br();                     // 右下角
	rectPoints[3] = cv::Point(boundBox.x, boundBox.y + boundBox.height); // 左下角

	cv::Mat drawLine = cv::Mat::zeros(src.size(), CV_8UC1);
	cv::rectangle(drawLine, boundBox, (255), 1);
	cv::line(drawLine, endPoint.first, endPoint.second, cv::Scalar(255), 1);
	for (int i = 0; i < 4; ++i) {
		cv::Point intersection = calculateLineIntersection(p1, p2, rectPoints[i], rectPoints[(i + 1) % 4]);
		if (intersection.x >= boundBox.x && intersection.x <= boundBox.x + boundBox.width && intersection.y >= boundBox.y && intersection.y <= boundBox.y + boundBox.height)
		{
			interPoints.push_back(intersection);
			//cv::circle(drawLine, intersection, 3, 255, 3);
		}

	}
	//cv::imshow("rect inter", drawLine);
	//cv::waitKey(0);

	return 1;
}

std::vector<cv::Point> ImageProcess::iterateLine(std::pair<cv::Point, cv::Point>& lineEdgePoints, int nStride)
{
	std::vector<cv::Point> vecPointsOnLine; // 存储最终符合条件的点

	cv::Point start = lineEdgePoints.first;
	cv::Point end = lineEdgePoints.second;

	// 计算直线的增量
	int dx = end.x - start.x;
	int dy = end.y - start.y;

	// 计算直线的长度
	float lineLength = std::sqrt(dx * dx + dy * dy);
	if (lineLength == 0) return { start }; // 如果起点和终点相同，返回起点

	// 处理步长的合法性
	if (nStride <= 0) {
		throw std::invalid_argument("stepLength must be positive.");
	}

	// 计算步数
	int steps = static_cast<int>(lineLength / nStride);

	// 计算单位步长
	float xStep = dx / lineLength * nStride; // 单位步长（x方向）
	float yStep = dy / lineLength * nStride; // 单位步长（y方向）

	// 初始化起点
	float x = static_cast<float>(start.x);
	float y = static_cast<float>(start.y);

	// 遍历直线上的每个点
	for (int i = 0; i <= steps; ++i) {
		// 获取当前点
		cv::Point currentPoint(static_cast<int>(std::round(x)), static_cast<int>(std::round(y)));
		vecPointsOnLine.push_back(currentPoint);

		// 更新点的坐标
		x += xStep;
		y += yStep;
	}

	// 确保最后一个点被添加
	if (vecPointsOnLine.empty() || vecPointsOnLine.back() != end) {
		vecPointsOnLine.push_back(end);
	}

	return vecPointsOnLine;
	//std::vector<cv::Point> vecPointsOnLine; // 存储最终符合条件的点

	//cv::Point start = lineEdgePoints.first;
	//cv::Point end = lineEdgePoints.second;

	//// 计算直线的增量
	//int dx = end.x - start.x;
	//int dy = end.y - start.y;

	//// 计算直线的长度
	//int steps = std::max(std::abs(dx), std::abs(dy)); // 直线像素点的数量
	//float xStep = dx / static_cast<float>(steps) * nStride;     // 单位步长（x方向）
	//float yStep = dy / static_cast<float>(steps) * nStride;     // 单位步长（y方向）

	//// 初始化起点
	//float x = static_cast<float>(start.x);
	//float y = static_cast<float>(start.y);

	//// 遍历直线上的每个点
	//for (int i = 0; i <= steps; i += nStride)
	//{
	//	// 获取当前点
	//	cv::Point currentPoint(static_cast<int>(std::round(x)), static_cast<int>(std::round(y)));
	//	vecPointsOnLine.push_back(currentPoint);

	//	// 更新点的坐标
	//	x += xStep;
	//	y += yStep;
	//}

	//return vecPointsOnLine;
}

int ImageProcess::getContourIntersectionPoint(cv::Mat& src, std::vector<std::vector<cv::Point>>& contour, std::vector<std::pair<cv::Point, cv::Point>>& vEndPoints,
	std::vector<std::vector<cv::Point>>& interPoints)
{
	//cv::Mat drawContours = cv::Mat::zeros(src.size(), CV_8UC1);
	//cv::drawContours(drawContours, contour, 0, cv::Scalar(255), 1);
	//for (int i = 0; i < vEndPoints.size(); i += 5)
	//{
	//	cv::Mat drawLine = cv::Mat::zeros(src.size(), CV_8UC1);
	//	
	//	auto endPointPair = vEndPoints[i];
	//	cv::line(drawLine, endPointPair.first, endPointPair.second, cv::Scalar(255), 1);

	//	cv::Mat inter;
	//	cv::bitwise_and(drawContours, drawLine, inter);
	//	
	//	std::vector<cv::Point> interPoint;
	//	getSkelPoints(inter, interPoint, 255);
	//	
	//	//removeDuplicatePoints(interPoint);

	//	interPoints.push_back(interPoint);

	//	cv::drawContours(drawLine, contour, 0, cv::Scalar(255), 1);
	//	for (auto& point : interPoint)
	//	{
	//		cv::circle(drawLine, point, 3, 255, 3);
	//	}
	//	
	//	cv::imshow("inter", drawLine);
	//	cv::waitKey(0);
	//}

	//cv::Rect contourBoundRect = cv::boundingRect(contour[0]);  // 最小包围矩形

	//for (int i = 0; i < vEndPoints.size(); i++)
	//{
	//	auto endPointPair = vEndPoints[i];
	//	std::vector<cv::Point> rectInterPoints = std::vector<cv::Point>{ endPointPair.first, endPointPair.second };
	//	std::vector<cv::Point> contourInterPoints;
	//	//getLineRectIntersectionPoint(src, contourBoundRect, endPointPair, rectInterPoints);

	//	if (rectInterPoints.size() >= 2)
	//	{
	//		cv::LineIterator lineIt(rectInterPoints[0], rectInterPoints[1], 8, true);
	//		cv::Point closestPoint;
	//		double dbMinDist = 10000.0;

	//		for (int j = 0; j < lineIt.count; ++j, ++lineIt)
	//		{
	//			cv::Point currPoint = lineIt.pos();
	//			double dbDist = cv::pointPolygonTest(contour[0], currPoint, true);
	//			if (dbDist >= 0)
	//			{
	//				contourInterPoints.push_back(currPoint);
	//			}
	//			else
	//			{
	//				if (std::abs(dbDist) < dbMinDist)
	//				{
	//					dbMinDist = std::abs(dbDist);
	//					closestPoint = currPoint;
	//				}
	//			}
	//		}

	//		//removeDuplicatePoints(contourInterPoints, 2.0f);

	//		std::vector<cv::Point> vSelectedInterPoints;

	//		if (contourInterPoints.empty())
	//			vSelectedInterPoints = { closestPoint, closestPoint };

	//		else if (contourInterPoints.size() == 1)
	//			vSelectedInterPoints = { contourInterPoints[0], contourInterPoints[0] };

	//		else if (contourInterPoints.size() >= 2)
	//			vSelectedInterPoints = { contourInterPoints[0], contourInterPoints.back() };

	//		//else if (contourInterPoints.size() < 2)
	//		//	continue;

	//		//cv::Mat drawLine = cv::Mat::zeros(src.size(), CV_8UC1);
	//		//cv::drawContours(drawLine, contour, 0, cv::Scalar(255), 1.8);
	//		//cv::line(drawLine, endPointPair.first, endPointPair.second, cv::Scalar(255), 1.8);
	//		//for (auto& point : vSelectedInterPoints)
	//		//{
	//		//	cv::circle(drawLine, point, 3, 255, 3);
	//		//}

	//		//cv::imshow("inter", drawLine);
	//		//cv::waitKey(0);
	//		interPoints.push_back(vSelectedInterPoints);	
	//	}
	//}
		//cv::Mat drawContours = cv::Mat::zeros(src.size(), CV_8UC1);
	//cv::drawContours(drawContours, contour, 0, cv::Scalar(255), 1);
	//for (int i = 0; i < vEndPoints.size(); i += 5)
	//{
	//	cv::Mat drawLine = cv::Mat::zeros(src.size(), CV_8UC1);
	//	
	//	auto endPointPair = vEndPoints[i];
	//	cv::line(drawLine, endPointPair.first, endPointPair.second, cv::Scalar(255), 1);

	//	cv::Mat inter;
	//	cv::bitwise_and(drawContours, drawLine, inter);
	//	
	//	std::vector<cv::Point> interPoint;
	//	getSkelPoints(inter, interPoint, 255);
	//	
	//	//removeDuplicatePoints(interPoint);

	//	interPoints.push_back(interPoint);

	//	cv::drawContours(drawLine, contour, 0, cv::Scalar(255), 1);
	//	for (auto& point : interPoint)
	//	{
	//		cv::circle(drawLine, point, 3, 255, 3);
	//	}
	//	
	//	cv::imshow("inter", drawLine);
	//	cv::waitKey(0);
	//}

	cv::Rect contourBoundRect = cv::boundingRect(contour[0]);  // 最小包围矩形

	for (int i = 0; i < vEndPoints.size(); i++)
	{
		auto endPointPair = vEndPoints[i];
		std::vector<cv::Point> rectInterPoints = std::vector<cv::Point>{ endPointPair.first, endPointPair.second };
		std::vector<cv::Point> contourInterPoints;
		//getLineRectIntersectionPoint(src, contourBoundRect, endPointPair, rectInterPoints);

		if (rectInterPoints.size() >= 2)
		{
			std::pair<cv::Point, cv::Point> pairLineEdgePoints{ rectInterPoints[0], rectInterPoints[1] };
			std::vector<cv::Point> vecPointsOnLine = iterateLine(pairLineEdgePoints, 4);

			//std::vector<cv::Point> vecBeginEdnPointsOnLine = std::vector<cv::Point>(vecPointsOnLine.begin(), vecPointsOnLine.begin() + 50);
			//vecBeginEdnPointsOnLine.insert(vecBeginEdnPointsOnLine.end(), vecPointsOnLine.end() - 50, vecPointsOnLine.end());
			
			double dMinDist = 10000.0;
			std::vector<double> vecPointToContourDist;
			cv::Point closestPoint;

			for (auto& currPoint : vecPointsOnLine)
			{
				if (!judgePointInBoundingRect(currPoint, contourBoundRect))
					continue;

				double dDist = cv::pointPolygonTest(contour[0], currPoint, true);
				vecPointToContourDist.push_back(dDist);
				if (dDist >= 0)
				{
					contourInterPoints.push_back(currPoint);
				}
				else
				{
					if (std::abs(dDist) < dMinDist)
					{
						dMinDist = std::abs(dDist);
						closestPoint = currPoint;
					}
				}
			}

			//cv::LineIterator lineIt(rectInterPoints[0], rectInterPoints[1], 8, true);

			//for (int j = 0; j < lineIt.count; ++j, ++lineIt)
			//{
			//	cv::Point currPoint = lineIt.pos();
			//	double dbDist = cv::pointPolygonTest(contour[0], currPoint, true);
			//	if (dbDist >= 0)
			//	{
			//		contourInterPoints.push_back(currPoint);
			//	}
			//	else
			//	{
			//		if (std::abs(dbDist) < dbMinDist)
			//		{
			//			dbMinDist = std::abs(dbDist);
			//			closestPoint = currPoint;
			//		}
			//	}
			//}

			//removeDuplicatePoints(contourInterPoints, 2.0f);

			std::vector<cv::Point> vSelectedInterPoints;

			if (contourInterPoints.empty())
				vSelectedInterPoints = { closestPoint, closestPoint };

			else if (contourInterPoints.size() == 1)
				vSelectedInterPoints = { contourInterPoints[0], contourInterPoints[0] };

			else if (contourInterPoints.size() >= 2)
				vSelectedInterPoints = { contourInterPoints[0], contourInterPoints.back() };

			//else if (contourInterPoints.size() < 2)
			//	continue;

			//cv::Mat drawLine = cv::Mat::zeros(src.size(), CV_8UC1);
			//cv::drawContours(drawLine, contour, 0, cv::Scalar(255), 1);
			//cv::line(drawLine, endPointPair.first, endPointPair.second, cv::Scalar(255), 1);
			//for (auto& point : vSelectedInterPoints)
			//{
			//	cv::circle(drawLine, point, 3, 255, 3);
			//}

			//cv::imshow("inter", drawLine);
			//cv::waitKey(0);
			interPoints.push_back(vSelectedInterPoints);
		}
	}

	return 1;
}

int ImageProcess::removeDuplicatePoints(std::vector<cv::Point>& srcPoints)
{
	for (auto it1 = srcPoints.begin(); it1 != srcPoints.end(); it1++)
	{
		for (auto it2 = it1 + 1; it2 != srcPoints.end(); )
		{
			cv::Point v1 = *it1 - *it2;
			if ((std::abs(v1.x) == 1.0f && v1.y == 0.0f) || (v1.x == 0.0f && std::abs(v1.y) == 1.0f))
			{
				it2 = srcPoints.erase(it2);
			}
			else
			{
				++it2;
			}
		}
	}
	return 1;
}

int ImageProcess::removeDuplicatePoints(std::vector<cv::Point>& srcPoints, float distThresh)
{
	for (auto it1 = srcPoints.begin(); it1 != srcPoints.end(); it1++)
	{
		for (auto it2 = it1 + 1; it2 != srcPoints.end(); )
		{
			cv::Point v1 = *it1 - *it2;
			if (sqrt(static_cast<float>(v1.x * v1.x + v1.y * v1.y)) < distThresh)
			{
				it2 = srcPoints.erase(it2);
			}
			else
			{
				++it2;
			}
		}
	}
	return 1;
}

std::vector<cv::Point> ImageProcess::mapPointsBackToOriginal(const cv::Mat& originalImage, const std::vector<cv::Point>& scaledPoints)
{
	std::vector<cv::Point> originalPoints;

	// 获取原始图像的尺寸
	int originalWidth = originalImage.cols;
	int originalHeight = originalImage.rows;

	// 计算缩放比例
	double scaleX = static_cast<double>(originalWidth) / 256.0;
	double scaleY = static_cast<double>(originalHeight) / 256.0;

	// 将缩放后的关键点映射回原始坐标
	for (const auto& point : scaledPoints) {
		cv::Point originalPoint;
		originalPoint.x = point.x * scaleX;
		originalPoint.y = point.y * scaleY;
		originalPoints.push_back(originalPoint);
	}

	return originalPoints;
}

cv::Point ImageProcess::calculateLineIntersection(const cv::Point& line1Pt1, const cv::Point& line1Pt2, const cv::Point& line2Pt1, const cv::Point& line2Pt2)
{
	double a1 = line1Pt2.y - line1Pt1.y;
	double b1 = line1Pt1.x - line1Pt2.x;
	double c1 = a1 * line1Pt1.x + b1 * line1Pt1.y;

	double a2 = line2Pt2.y - line2Pt1.y;
	double b2 = line2Pt1.x - line2Pt2.x;
	double c2 = a2 * line2Pt1.x + b2 * line2Pt1.y;

	double determinant = a1 * b2 - a2 * b1;

	if (determinant == 0) {
		// Lines are parallel, intersection doesn't exist
		return cv::Point(-1, -1);
	}
	else {
		int x = static_cast<int>((c1 * b2 - c2 * b1) / determinant);
		int y = static_cast<int>((a1 * c2 - a2 * c1) / determinant);
		return cv::Point(x, y);
	}
}

void ImageProcess::orderRotatedRectPoints(cv::Point2f pts[4], cv::Point2f dst[4])
{
	std::sort(pts, pts + 4, compareRectPoints);

	cv::Point2f topLeft = pts[0].y < pts[1].y ? pts[0] : pts[1];
	cv::Point2f bottomLeft = pts[0].y > pts[1].y ? pts[0] : pts[1];
	cv::Point2f topRight = pts[2].y < pts[3].y ? pts[2] : pts[3];
	cv::Point2f bottomRight = pts[2].y > pts[3].y ? pts[2] : pts[3];

	// Re-assign the sorted points to the destination array
	dst[0] = topLeft;
	dst[1] = topRight;
	dst[2] = bottomRight;
	dst[3] = bottomLeft;
}

std::pair<cv::Mat, cv::Mat> ImageProcess::affineTransform(cv::Mat src, int fixedSize[2])  // 仿射变换
{
	int targetBox[4] = { 0.f, 0.f , src.cols - 1.f, src.rows - 1.f }; // {w,h}
	std::vector<float> adjustValues = adjustImage(targetBox, fixedSize);
	float srcXMin = adjustValues[0];
	float srcYMin = adjustValues[1];
	float srcXMax = adjustValues[2];
	float srcYMax = adjustValues[3];

	float srcW = srcXMax - srcXMin;
	float srcH = srcYMax - srcYMin;
	 
	cv::Point2f srcCenter = { (srcXMin + srcXMax) / 2.0f,(srcYMin + srcYMax) / 2.0f };
	cv::Point2f srcPointOne = { srcCenter.x, srcCenter.y - (srcH / 2.0f) };
	cv::Point2f srcPointTwo = { srcCenter.x + (srcW / 2.0f), srcCenter.y };

	cv::Point2f dstCenter = { (fixedSize[1] - 1.0f) / 2.0f, (fixedSize[0] - 1.0f) / 2.0f };
	cv::Point2f dstPointOne = { (fixedSize[1] - 1.0f) / 2.0f, 0 };
	cv::Point2f dstPointTwo = { fixedSize[1] - 1.0f, (fixedSize[0] - 1) / 2.0f };

	cv::Point2f srcPoints[3] = { srcCenter, srcPointOne, srcPointTwo };
	cv::Point2f dstPoints[3] = { dstCenter, dstPointOne, dstPointTwo };

	cv::Mat originTransResize = cv::getAffineTransform(srcPoints, dstPoints);
	cv::Size wantedSize(fixedSize[1], fixedSize[0]);
	cv::Mat resizedImg;
	cv::warpAffine(src, resizedImg, originTransResize, wantedSize, cv::INTER_LINEAR);

	for (auto& point : dstPoints) 
	{
		point.x /= 4.0f;
		point.y /= 4.0f;
	}

	cv::Mat outputTransOrigin = cv::getAffineTransform(dstPoints, srcPoints);
	//auto a = outputTransOrigin.at<double>(0, 0);
	//auto b = outputTransOrigin.at<double>(0, 1);
	//auto c = outputTransOrigin.at<double>(0, 2);
	//auto d = outputTransOrigin.at<double>(1, 0);
	//auto e = outputTransOrigin.at<double>(1, 1);
	//auto f = outputTransOrigin.at<double>(1, 2);

	std::pair<cv::Mat, cv::Mat> dstImgAndTransPair = std::make_pair(resizedImg, outputTransOrigin);
	return dstImgAndTransPair;
}

std::pair<cv::Mat, cv::Mat> ImageProcess::affineTransform(cv::Mat src, int fixedSize[2], float scale)  //     任
{
	int targetBox[4] = { 0.f, 0.f , src.cols - 1.f, src.rows - 1.f }; // {w,h}
	std::vector<float> adjustValues = adjustImage(targetBox, fixedSize);
	float srcXMin = adjustValues[0];
	float srcYMin = adjustValues[1];
	float srcXMax = adjustValues[2];
	float srcYMax = adjustValues[3];

	float srcW = srcXMax - srcXMin;
	float srcH = srcYMax - srcYMin;

	cv::Point2f srcCenter = { (srcXMin + srcXMax) / 2.0f,(srcYMin + srcYMax) / 2.0f };
	cv::Point2f srcPointOne = { srcCenter.x, srcCenter.y - (srcH / 2.0f) };
	cv::Point2f srcPointTwo = { srcCenter.x + (srcW / 2.0f), srcCenter.y };

	if (scale)
	{
		srcW *= scale;
		srcH *= scale;
		srcPointOne = { srcCenter.x, srcCenter.y - (srcH / 2.0f) };
		srcPointTwo = { srcCenter.x + (srcW / 2.0f), srcCenter.y };
	}

	cv::Point2f dstCenter = { (fixedSize[1] - 1.0f) / 2.0f, (fixedSize[0] - 1.0f) / 2.0f };
	cv::Point2f dstPointOne = { (fixedSize[1] - 1.0f) / 2.0f, 0 };
	cv::Point2f dstPointTwo = { fixedSize[1] - 1.0f, (fixedSize[0] - 1) / 2.0f };

	cv::Point2f srcPoints[3] = { srcCenter, srcPointOne, srcPointTwo };
	cv::Point2f dstPoints[3] = { dstCenter, dstPointOne, dstPointTwo };

	cv::Mat originTransResize = cv::getAffineTransform(srcPoints, dstPoints);
	cv::Size wantedSize(fixedSize[1], fixedSize[0]);
	cv::Mat resizedImg;
	cv::warpAffine(src, resizedImg, originTransResize, wantedSize, cv::INTER_LINEAR);

	for (auto& point : dstPoints)
	{
		point.x /= 4.0f;
		point.y /= 4.0f;
	}

	cv::Mat outputTransOrigin = cv::getAffineTransform(dstPoints, srcPoints);
	//auto a = outputTransOrigin.at<double>(0, 0);
	//auto b = outputTransOrigin.at<double>(0, 1);
	//auto c = outputTransOrigin.at<double>(0, 2);
	//auto d = outputTransOrigin.at<double>(1, 0);
	//auto e = outputTransOrigin.at<double>(1, 1);
	//auto f = outputTransOrigin.at<double>(1, 2);

	std::pair<cv::Mat, cv::Mat> dstImgAndTransPair = std::make_pair(resizedImg, outputTransOrigin);
	return dstImgAndTransPair;
}

std::vector<float> ImageProcess::adjustImage(int targetBox[4], int fixedSize[2])  // 通过增加w或者h的方式保证输入图片的长宽比固定 targetBox = {xmin, ymin, w, h} fixedSize = {h, w}
{
	float xMin = targetBox[0];
	float yMin = targetBox[1];
	float w = targetBox[2];
	float h = targetBox[3];
	float xMax = xMin + w;
	float yMax = yMin + h;

	float hwRatio = static_cast<float>(fixedSize[0]) / static_cast<float> (fixedSize[1]);

	if (h / w > hwRatio)   // 需要在w方向padding
	{
		float wExpeted = h / hwRatio;
		float wPad = (wExpeted - w) / 2;
		xMin = xMin - wPad;
		xMax = xMax + wPad;
	}
	else // 需要在h方向padding
	{
		float hExpeted = w * hwRatio;
		float hPad = (hExpeted - h) / 2;
		yMin = yMin - hPad;
		yMax = yMax + hPad;
	}

	std::vector<float> adjustValues = { xMin,yMin, xMax, yMax };
	return adjustValues;
}

std::vector<cv::Point> ImageProcess::affinePoints(std::vector<cv::Point2f> originPoints, cv::Mat trans) 
{
	//cv::Mat pointsMat(3, 2, CV_64FC1);
	//cv::Mat result;
	//pointsMat.at<double>(0, 0) = originPoints[0].x;
	//pointsMat.at<double>(0, 1) = originPoints[1].x;
	//pointsMat.at<double>(1, 0) = originPoints[0].y;
	//pointsMat.at<double>(1, 1) = originPoints[1].y;
	//pointsMat.at<double>(2, 0) = 1.0f;
	//pointsMat.at<double>(2, 1) = 1.0f;

	//
	//result = trans * pointsMat;

	//cv::Point transedPointOne, transedPointTwo;
	//transedPointOne.x = result.at<double>(0, 0);
	//transedPointOne.y = result.at<double>(1, 0);
	//transedPointTwo.x = result.at<double>(0, 1);
	//transedPointTwo.y = result.at<double>(1, 1);
	//std::vector<cv::Point> transedPoints = { transedPointOne, transedPointTwo };
	//return transedPoints;
	int numPoints = originPoints.size();
	cv::Mat pointsMat(3, numPoints, CV_64FC1);
	cv::Mat result;
	for (int col = 0; col < pointsMat.cols; col++)
	{
		pointsMat.at<double>(0, col) = originPoints[col].x;
		pointsMat.at<double>(1, col) = originPoints[col].y;
		pointsMat.at<double>(2, col) = 1.0f;
	}
	result = trans * pointsMat;

	std::vector<cv::Point> transedPoints;
	for (int c = 0; c < result.cols; c++)
	{
		cv::Point tempPoint;
		tempPoint.x = result.at<double>(0, c);
		tempPoint.y = result.at<double>(1, c);
		transedPoints.push_back(tempPoint);
	}
	return transedPoints;
}

int ImageProcess::ivsAndPWPostProcess(cv::Mat& src, std::vector<cv::Mat>& vMasks, std::map<std::string, std::vector<float>>& values, std::map<std::string, cv::Mat>& resultPics)
{
	std::vector<std::vector<cv::Point>> contours;
	std::vector<cv::Vec4i> hierarchy;
	cv::Mat ivsSkeleton, pwSkeleton;
	cv::Mat ivsMask, pwMask;

	ivsMask = vMasks[0];
	pwMask = vMasks[1];

	findMaxAreaConnected(ivsMask, ivsMask);
	findMaxAreaConnected(pwMask, pwMask);

	cv::resize(ivsMask, ivsMask, src.size(), 0.0, 0.0, cv::INTER_NEAREST);
	cv::resize(pwMask, pwMask, src.size(), 0.0, 0.0, cv::INTER_NEAREST);

	cv::Mat displayContours = cv::Mat::zeros(ivsMask.size(), CV_32SC1);
	ivsMask.convertTo(ivsMask, CV_8UC1);
	pwMask.convertTo(pwMask, CV_8UC1);

	cv::Mat binaryMask;
	//cv::imshow("PW Mask", pwMask);
	//cv::imshow("IVS Mask", ivsMask);
	//cv::waitKey(0);

	getSkeletonizeMask(ivsMask, ivsSkeleton);
	getSkeletonizeMask(pwMask, pwSkeleton);

	std::vector<cv::Point> ivsPoints, pwPoints;

	// for test
	//cv::imshow("PW skeleton", pwSkeleton);
	//cv::imshow("IVS skeleton", ivsSkeleton);
	//cv::waitKey(0);

	getSkelPoints(ivsSkeleton, ivsPoints, 255);
	getSkelPoints(pwSkeleton, pwPoints, 255);

	if (pwPoints.size() == 0 || ivsPoints.size() == 0)
	{
		return 0;
	}

	removeDuplicatePoints(ivsPoints);
	removeDuplicatePoints(pwPoints);

	pointSortByX(ivsPoints);
	pointSortByX(pwPoints);

	std::vector<cv::Point> midLine;
	std::vector<cv::Point2f> smoothedMidLine, smoothedIVSLine, smoothedPWLine;
	windowAverageCurve(ivsPoints, smoothedIVSLine, 5);
	windowAverageCurve(pwPoints, smoothedPWLine, 5);
	getLVMidLine(smoothedIVSLine, smoothedPWLine, midLine);
	windowAverageCurve(midLine, smoothedMidLine, 9);

	cv::Mat drawImage = src.clone();

	// for test
	//for (auto& point : ivsPoints)
	//	cv::circle(drawImage, point, 2, cv::Scalar(0, 0, 255), 2);
	//cv::imshow("midline", drawImage);
	//cv::waitKey(0);


	std::vector<std::pair<float, float>> vNormals;
	calcNormals(smoothedMidLine, vNormals);

	std::vector<std::pair<cv::Point, cv::Point>> endPointLists;
	extendNormalsOnImage(drawImage, smoothedMidLine, vNormals, endPointLists);

	//for (auto endPoint : endPointLists)
	//{
	//	cv::line(drawImage, endPoint.first, endPoint.second, cv::Scalar(0, 0, 255), 1);
	//}
	//cv::imshow("midline", drawImage);
	//cv::waitKey(0);

	std::vector<std::vector<cv::Point>> ivsContours, pwContours;
	std::vector<std::vector<cv::Point>> ivsInterPoints, pwInterPoints;
	cv::findContours(ivsMask, ivsContours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
	cv::findContours(pwMask, pwContours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
	filterSmallContour(ivsContours);
	filterSmallContour(pwContours);

	// for test
	//cv::drawContours(drawImage, ivsContours, -1, cv::Scalar(0, 0, 255), 2);
	//for (auto& point : smoothedIVSLine)
	//	cv::circle(drawImage, point, 2, cv::Scalar(0, 0, 255), 2);
	//cv::imshow("midline", drawImage);
	//cv::waitKey(0);

	getContourIntersectionPoint(src, ivsContours, endPointLists, ivsInterPoints);
	getContourIntersectionPoint(src, pwContours, endPointLists, pwInterPoints);
	if (ivsInterPoints.size() == 0 || pwInterPoints.size() == 0)
	{
		return 0;
	}

	// for test
	//cv::drawContours(drawImage, ivsContours, -1, cv::Scalar(0, 0, 255), 2);
	//cv::drawContours(drawImage, pwContours, -1, cv::Scalar(0, 255, 0), 2);
	//for (auto& point : smoothedMidLine)
	//	cv::circle(drawImage, point, 2, cv::Scalar(255, 0, 0), 2);
	//for (auto& point : smoothedIVSLine)
	//	cv::circle(drawImage, point, 2, cv::Scalar(0, 0, 255), 2);
	//for (auto& point : smoothedPWLine)
	//	cv::circle(drawImage, point, 2, cv::Scalar(0, 0, 255), 2);
	//for (auto endPoint : endPointLists)
	//{
	//	cv::line(drawImage, endPoint.first, endPoint.second, cv::Scalar(0, 0, 255), 1);
	//}
	//cv::imshow("midline", drawImage);
	//cv::waitKey(0);

	std::vector<cv::Point> ivsLine, pwLine, lvidLine;
	float fIVSDist = getIVSThickness(ivsInterPoints, ivsLine);
	float fPWDist = getPWStructureThickness(pwInterPoints, pwLine);
	float fLVIDDist = getLVIDStructureThickness(ivsInterPoints, pwInterPoints, lvidLine);

	//cv::drawContours(drawImage, ivsContours, -1, cv::Scalar(0, 0, 255), 2);
	//cv::drawContours(drawImage, pwContours, -1, cv::Scalar(0, 255, 0), 2);
	//for (auto& point : smoothedMidLine)
	//	cv::circle(drawImage, point, 2, cv::Scalar(255, 0, 0), 2);
	//for (auto& point : smoothedIVSLine)
	//	cv::circle(drawImage, point, 2, cv::Scalar(0, 0, 255), 2);
	//for (auto& point : smoothedPWLine)
	//	cv::circle(drawImage, point, 2, cv::Scalar(0, 0, 255), 2);

	cv::Mat drawImageIsv = src.clone();
	cv::Mat drawImagePw = src.clone();
	cv::Mat drawImageLvid = src.clone();

	cv::line(drawImageIsv, ivsLine[0], ivsLine[1], cv::Scalar(0, 255, 0), 2);  // 绿色
	float ivsDst = ParamsAssessUtils::calcLineDist(ivsLine);

	cv::line(drawImagePw, pwLine[0], pwLine[1], cv::Scalar(0, 0, 255), 2);
	float pwDst = ParamsAssessUtils::calcLineDist(pwLine);

	cv::line(drawImageLvid, lvidLine[0], lvidLine[1], cv::Scalar(255, 0, 0), 2);
	float lvidDst = ParamsAssessUtils::calcLineDist(lvidLine);

	values.insert({ "LVDd", std::vector<float>{lvidDst} });
	values.insert({ "IVSTd", std::vector<float>{ivsDst} });
	values.insert({ "LVPWTd", std::vector<float>{pwDst } });
	resultPics.insert({ "LVDd", drawImageLvid });
	resultPics.insert({ "IVSTd", drawImageIsv });
	resultPics.insert({ "LVPWTd", drawImagePw });
	//values["LVPWd"] = pwDst;

	//cv::imshow("midline", drawImageLvid);
	//cv::imshow("midline1", drawImageIsv);
	//cv::imshow("midline2", drawImagePw);
	//cv::waitKey(0);
	return 1;
}

float ImageProcess::getPWStructureThickness(std::vector<std::vector<cv::Point>>& interPoints, std::vector<cv::Point>& resultLine)
{
	std::vector<float> vDists;
	ParamsAssessUtils::calcLinesDistance(interPoints, vDists);

	int segmentIdx2 = static_cast<int>(interPoints.size() * 2.0f / 3.0f);

	resultLine = interPoints[segmentIdx2];
	return vDists[segmentIdx2];
}

float ImageProcess::getIVSThickness(std::vector<std::vector<cv::Point>>& interPoints, std::vector<cv::Point>& resultLine)
{
	std::vector<float> vDists;
	ParamsAssessUtils::calcLinesDistance(interPoints, vDists);

	//int segment1Range = static_cast<int>(interPoints.size() / 3.0f);
	//auto distMaxElem = std::max_element(vDists.begin(), vDists.begin() + segment1Range);
	//int maxIdx = std::distance(vDists.begin(), distMaxElem);
	//float maxDist = *distMaxElem;

	int segmentIdx2 = static_cast<int>(interPoints.size() * 0.5f);
	if (segmentIdx2 == 0)
	{
		segmentIdx2 += 1;
	}
	auto seg2MaxElem = std::max_element(vDists.begin() + segmentIdx2 - 1, vDists.end());
	int seg2MaxIdx = std::distance(vDists.begin(), seg2MaxElem);
	std::vector<cv::Point> closestDistPoints = interPoints[seg2MaxIdx];
	float closestDist = vDists[seg2MaxIdx];
	resultLine = closestDistPoints;
	return closestDist;

	//if (maxDist - closestDist > 0.3f * maxDist)
	//{
	//	resultLine = closestDistPoints;
	//	return closestDist;
	//}
	//else
	//{
	//	resultLine = interPoints[maxIdx];
	//	return maxDist;
	//}
}

std::vector<cv::Point> ImageProcess::getLVIDPoint(std::vector<cv::Point>& ivsPoint, std::vector<cv::Point>& pwPoint)
{
	cv::Point ivsBottomPoint = ivsPoint[0].y > ivsPoint[1].y ? ivsPoint[0] : ivsPoint[1];
	cv::Point pwUpPoint = pwPoint[0].y < pwPoint[1].y ? pwPoint[0] : pwPoint[1];

	std::vector<cv::Point> vLVIDPoint{ ivsBottomPoint, pwUpPoint };
	return vLVIDPoint;
}

float ImageProcess::getLVIDStructureThickness(std::vector<std::vector<cv::Point>>& ivsInterPoints, std::vector<std::vector<cv::Point>>& pwInterPoints, std::vector<cv::Point>& resultLine)
{
	int numPointSample = std::min(pwInterPoints.size(), ivsInterPoints.size());
	std::vector<std::vector<cv::Point>> vLVIDPoints;
	std::vector<float> vDists;
	for (int i = 0; i < numPointSample; i++)
	{
		std::vector<cv::Point> currIVSInterPoint = ivsInterPoints[i];
		std::vector<cv::Point> currPWInterPoint = pwInterPoints[i];
		std::vector<cv::Point> currLVIDPoint = getLVIDPoint(currIVSInterPoint, currPWInterPoint);
		vLVIDPoints.push_back(currLVIDPoint);
	}
	ParamsAssessUtils::calcLinesDistance(vLVIDPoints, vDists);

	int startIdx = static_cast<int>(vLVIDPoints.size() / 3.0f);
	int endIdx = static_cast<int>(vLVIDPoints.size() * 7.0f / 10.0f);
	auto maxDistElem = std::max_element(vDists.begin() + startIdx, vDists.begin() + endIdx);
	int maxIdx = std::distance(vDists.begin(), maxDistElem);

	resultLine = vLVIDPoints[maxIdx];

	return *maxDistElem;
}

int ImageProcess::ladPostProcess(cv::Mat& src, cv::Mat& laMask, cv::Mat& avMask, std::map<std::string, std::vector<float>>& values, std::map<std::string, cv::Mat>& resultPics)
{
	////=======================左心房部分====================////
	std::vector<std::vector<cv::Point>> laMaskContours;
	laMask.convertTo(laMask, CV_8UC1);
	cv::resize(laMask, laMask, src.size());
	//cv::imshow("la mask", laMask);
	//cv::waitKey(0);
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
	//	//cv::cvtColor(laMask, drawLine, cv::COLOR_GRAY2BGR);
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
	cv::Mat drawLine = src.clone();
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
	//cv::imshow("av mask", avMask);
	//cv::waitKey(0);
	avMask.convertTo(avMask, CV_8UC1);
	cv::resize(avMask, avMask, src.size());
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

	drawLine = src.clone();
	//cv::Size outputSize(256, 256);
	//cv::resize(drawLine, drawLine, outputSize);

	//cv::circle(drawLine, avPoint, 5, cv::Scalar(0, 255, 255), -1);
	cv::line(drawLine, laavPoints[0], laavPoints[1], cv::Scalar(0, 255, 0), 2);  // 应该是可以在drawImage上画的
	float laadDst = ParamsAssessUtils::calcLineDist(laavPoints);

	values.insert({ "LAD", std::vector<float>{laadDst} });
	resultPics.insert({ "LAD", drawLine });

	//cv::imshow("test", drawLine);
	//cv::waitKey(0);

	return 1;
}

int ImageProcess::aadPostProcess(cv::Mat& src, cv::Mat& aadMask, std::map<std::string, std::vector<float>>& values, std::map<std::string, cv::Mat>& resultPics)
{
	std::vector<std::vector<cv::Point>> maskContours;
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
	if (!interceptPoints.empty())
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

int ImageProcess::asdAndSJDAPostProcess(cv::Mat& src, cv::Mat& asdAndsjdMask, std::map<std::string, std::vector<float>>& values, std::map<std::string, cv::Mat>& resultPics)
{
	std::vector<std::vector<cv::Point>> maskContours;
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
	std::vector<std::pair<double, size_t>> UpperPeakDistancesPair = ParamsAssessUtils::findLocalMaximumPair(smoothedUpperDistances);

	//从现有平稳点中挑出极小值点或删去极大值点
	std::vector<float> GradientUpperDistances = ParamsAssessUtils::gradientOneDimension(smoothedUpperDistances);
	std::vector<size_t> SteadyUpperPointsIdx = ParamsAssessUtils::findIndices(GradientUpperDistances, 0.11f);  // 如果没有找到小于阈值的点会报错
	std::vector<size_t> copySteadyUpperPointsIdx(SteadyUpperPointsIdx);

	for (size_t index : SteadyUpperPointsIdx)
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

std::pair<int, float> ImageProcess::getAADDist(std::vector<std::vector<cv::Point>>& vInterPoints)
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

int ImageProcess::polyfit(const std::vector<cv::Point>& points, int degree, std::vector<double>& coefficients)
{
	int N = points.size();
	int num_coeffs = degree + 1;

	// 构造矩阵 X 和向量 Y
	cv::Mat1d X(N, num_coeffs);
	cv::Mat1d Y(N, 1);
	for (int i = 0; i < N; ++i) {
		double xi = points[i].x;
		double yi = points[i].y;
		Y(i, 0) = yi;

		for (int j = 0; j < num_coeffs; ++j) {
			X(i, j) = std::pow(xi, j);
		}
	}

	// 使用最小二乘法计算 (X^T * X)^{-1} * X^T * Y
	cv::Mat1d XtX = X.t() * X;
	cv::Mat1d XtY = X.t() * Y;
	cv::Mat1d coeffs;
	cv::solve(XtX, XtY, coeffs, cv::DECOMP_SVD);

	// 将结果存储到 vector 中
	for (int i = 0; i < coeffs.rows; ++i) {
		coefficients.push_back(coeffs.at<double>(i, 0));
	}
	return 1;
}

int ImageProcess::findClosestPointOnLine(  // 计算点到直线的距离，并找到最小距离的点，视作这条直线与离散点的交点
	const cv::Point& lineStart, const cv::Point& lineEnd, const std::vector<cv::Point>& points, cv::Point& closestPoint)
{
	// 直线方程 Ax + By + C = 0 的系数
	float A = lineEnd.y - lineStart.y;
	float B = lineStart.x - lineEnd.x;
	float C = lineEnd.x * lineStart.y - lineStart.x * lineEnd.y;

	float minDistance = std::numeric_limits<float>::max();

	for (const auto& point : points) {
		// 计算点到直线的垂直距离
		float distance = std::abs(A * point.x + B * point.y + C) / std::sqrt(A * A + B * B);

		// 更新最小距离的点
		if (distance < minDistance) {
			minDistance = distance;
			closestPoint = point;
		}
	}

	return 1;
}

int ImageProcess::getPerpendicularLineEndpoints(  // 给定一点，计算垂线的端点
	const std::vector<double>& coeffs, const cv::Point& pt_target, std::vector<cv::Point>& endPoints, double length)
{
	double x_target = static_cast<double>(pt_target.x);
	double y_target = static_cast<double>(pt_target.y);

	// 计算多项式的导数并求切线斜率
	std::vector<double> derivative = ImageProcess::polyderivative(coeffs);
	double slope_tangent = ImageProcess::evaluatePolynomial(derivative, x_target);

	// 计算垂线的斜率
	double slope_perpendicular = -1.0 / slope_tangent;

	// 根据垂线斜率计算偏移量
	double dx = length / std::sqrt(1 + slope_perpendicular * slope_perpendicular);
	double dy = slope_perpendicular * dx;

	// 计算垂线的两个端点
	cv::Point pt1(x_target + dx, y_target + dy);
	cv::Point pt2(x_target - dx, y_target - dy);
	endPoints = { pt1, pt2 };

	return 1;
}

// 计算两条线段的交点
bool ImageProcess::lineIntersection(const cv::Point& p1, const cv::Point& p2, const cv::Point& p3, const cv::Point& p4, cv::Point& intersection) {
	// 线段 p1p2 与 p3p4 的交点公式
	float x1 = p1.x, y1 = p1.y;
	float x2 = p2.x, y2 = p2.y;
	float x3 = p3.x, y3 = p3.y;
	float x4 = p4.x, y4 = p4.y;

	float denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4);
	if (denom == 0) return false;  // 如果平行或重合，则没有交点

	float intersect_x = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom;
	float intersect_y = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom;

	// 判断交点是否在线段上
	if (std::min(x1, x2) <= intersect_x && intersect_x <= std::max(x1, x2) &&
		std::min(y1, y2) <= intersect_y && intersect_y <= std::max(y1, y2) &&
		std::min(x3, x4) <= intersect_x && intersect_x <= std::max(x3, x4) &&
		std::min(y3, y4) <= intersect_y && intersect_y <= std::max(y3, y4)) {
		intersection = cv::Point(static_cast<int>(intersect_x), static_cast<int>(intersect_y));
		return true;
	}
	return false;
}

int ImageProcess::findLineContourIntersections(const std::vector<cv::Point>& line, const std::vector<std::vector<cv::Point>>& maskContours, std::vector<cv::Point>& intersections) {

	// 遍历轮廓
	for (const auto& contour : maskContours) {
		// 遍历轮廓的每一条线段
		for (size_t i = 0; i < contour.size(); ++i) {
			cv::Point p1 = contour[i];
			cv::Point p2 = contour[(i + 1) % contour.size()];  // 下一点，如果是最后一条边则回到第一个点
			cv::Point intersection;

			// 检查直线与当前线段是否相交
			if (ImageProcess::lineIntersection(line[0], line[1], p1, p2, intersection)) {
				intersections.push_back(intersection);
			}
		}
	}
	return 1;
}

bool ImageProcess::isColorJudge(cv::Mat& src)
{
	if (src.channels() != 3) {
		return false; // 如果不是三通道图像，返回未知类型
	}

	// 分离RGB通道
	std::vector<cv::Mat> channels;
	cv::split(src, channels);

	// 检查是否为灰度图像（RGB通道值相同）
	cv::Mat diff1, diff2;
	cv::absdiff(channels[0], channels[1], diff1); // 比较R和G通道
	cv::absdiff(channels[0], channels[2], diff2); // 比较R和B通道

	double maxDiff1, maxDiff2;
	cv::minMaxLoc(diff1, nullptr, &maxDiff1);
	cv::minMaxLoc(diff2, nullptr, &maxDiff2);

	if (maxDiff1 < 10 && maxDiff2 < 10) { // 如果RGB通道值差异很小，认为是B超图像
		return false;
	}

	// 检查是否存在明显的彩色区域
	cv::Mat hsvImage;
	cv::cvtColor(src, hsvImage, cv::COLOR_BGR2HSV);

	// 定义彩色区域的HSV范围（例如红色和蓝色）
	cv::Mat redMask1, redMask2, blueMask;
	cv::inRange(hsvImage, cv::Scalar(0, 50, 50), cv::Scalar(10, 255, 255), redMask1);
	cv::inRange(hsvImage, cv::Scalar(170, 50, 50), cv::Scalar(180, 255, 255), redMask2);
	cv::inRange(hsvImage, cv::Scalar(100, 50, 50), cv::Scalar(130, 255, 255), blueMask);

	cv::Mat colorMask = redMask1 | redMask2 | blueMask;
	double colorArea = cv::countNonZero(colorMask);

	if (colorArea > 0.01 * src.total()) { // 如果彩色区域超过一定比例，认为是彩超图像
		return true;
	}
	
	return false;
}

QImage ImageProcess::adjustContrast(const QImage& image, double alpha, int beta)
{
	QImage newImage = image.copy(); // 创建新图像以存储结果

	for (int y = 0; y < image.height(); ++y) {
		for (int x = 0; x < image.width(); ++x) {
			QColor pixelColor = image.pixelColor(x, y);
			// 调整每个颜色分量
			int red = qBound(0, static_cast<int>(alpha * pixelColor.red() + beta), 255);
			int green = qBound(0, static_cast<int>(alpha * pixelColor.green() + beta), 255);
			int blue = qBound(0, static_cast<int>(alpha * pixelColor.blue() + beta), 255);
			newImage.setPixelColor(x, y, QColor(red, green, blue));
		}
	}

	return newImage;
}

int ImageProcess::judgePointInBoundingRect(cv::Point& point, cv::Rect& rect)
{
	if (point.x < 0 || point.y < 0)
		return 0;

	int minX = rect.x;
	int minY = rect.y;
	int w = rect.width;
	int h = rect.height;

	int maxX = minX + w;
	int maxY = minY + h;

	if ((point.x >= minX && point.x <= maxX) && (point.y >= minY && point.y <= maxY))
		return 1;

	return 0;
}
