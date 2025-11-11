#pragma once
#include <QString>
#include <iostream>
#include <memory>
#include <opencv2/opencv.hpp>

struct ScaleInfo
{
	float length = -10000.0f;
	std::string unit = "";
	float unitPositionY = -10000.0f;
	float fPixelPerUnit = -10000.0f;
	float fSpecScaleRange = -10000.0f;

	void clear()
	{
		length = -10000.0f;
		unit = "";
		unitPositionY = -10000.0f;
		fPixelPerUnit = -10000.0f;
		fSpecScaleRange = -10000.0f;
	}
};

struct TextInfo
{
	std::string text = "";
	cv::Rect boundingBox = cv::Rect();
	float distInfo = -10000.0f;
	std::string unit = "";

	void clear()
	{
		text = "";
		boundingBox = cv::Rect();
		distInfo = -10000.0f;
		unit = "";
	}
};

struct PatientInfo
{
	std::string patientName;
	std::string patientID;

	void clear()
	{
		patientName.clear();
		patientID.clear();
	}
};

struct ModeInfo
{
	std::string mode = "";
	std::string specMode = "";

	bool bIsColorMode = false;

	void clear()
	{
		mode.clear();
		specMode.clear();
		bIsColorMode = false;
	}
};

struct RoIScaleInfo
{
	float fRadius = -10000.0f;
	cv::Rect roiRect;
	cv::Rect specScaleRect;

	void clear()
	{
		fRadius = -10000.0f;
		roiRect = cv::Rect();
		specScaleRect = cv::Rect();
	}
};

struct Object
{
    cv::Rect rect;
    int label;
    float conf;
};

struct PeakInfo {
	int index;
	float value;
	float width;
	float prominence;
};

struct FloatInferDeleter
{
	FloatInferDeleter() = default;

	FloatInferDeleter(const FloatInferDeleter&) = default;
	FloatInferDeleter& operator=(const FloatInferDeleter&) = default;
	FloatInferDeleter(FloatInferDeleter&&) = default;
	FloatInferDeleter& operator=(FloatInferDeleter&&) = default;

	template <typename T>
	void operator()(T* obj) const {
		delete[] obj;
	}
};

typedef std::shared_ptr<float[]> floatArrayPtr;

struct ParamData {
	QString parameter;
	QString value;
	QString unit;
};

struct ViewClsResult 
{
	QString strViewName;
	float fQualityScore = -10000.0f;
	bool bIsSwitch = false;

	void clear()
	{
		strViewName.clear();
		fQualityScore = -10000.0f;
		bIsSwitch = false;
	}

	bool isEmpty()
	{
		return strViewName.isEmpty() && fQualityScore == -10000.0f;
	}
};