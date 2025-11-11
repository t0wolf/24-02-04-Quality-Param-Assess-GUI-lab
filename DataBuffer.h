#pragma once

#include <QObject>
#include <QVector>
#include <QMutex>
#include "type_define.h"

class DataBuffer
{
	//Q_OBJECT

public:
	DataBuffer();
	~DataBuffer();

	void setData(RoIScaleInfo& roiScaleInfo);

	void getData(RoIScaleInfo& roiScaleInfo);

	bool checkRoIDataUpdate()
	{
		QMutexLocker locker(&m_dataBufferMutex);
		return m_isRoIDataUpdate;
	}

private:
	QMutex m_dataBufferMutex;
	RoIScaleInfo m_roiScaleInfoBuffer;
	cv::Mat m_imageBuffer;

	bool m_isRoIDataUpdate;
};
