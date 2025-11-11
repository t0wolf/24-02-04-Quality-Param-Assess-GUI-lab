#pragma once
#include <QWidget>
#include <QObject>
#include <QThread>
#include <QMutex>
#include <QVariant>
#include <Qdebug>

#include "DataBuffer.h"

#include "type_conversion.h"
#include "type_define.h"
#include "type_registering.h"
#include "general_utils.h"
#include "quality_control/roi_detection.h"


class ROIScaleDetThread : public QThread
{
	Q_OBJECT
public:
	ROIScaleDetThread(QObject* parent = nullptr, DataBuffer* roiDataBuffer = nullptr);

	void run();

	int inputVideoFrame(cv::Mat& frame);

	~ROIScaleDetThread()
	{
		exitThread();
	}

	void exitThread()
	{
		this->requestInterruption();
		this->quit();
		this->wait();
	}

signals:
	void sigROIScaleAvailable(QVariant qROIScaleInfo);

public slots:
	void setVideoFrame(const QImage frame);

private:
	int doROIDetInfer(cv::Mat& currFrame, std::vector<Object>& vObjects);

	int parseRoiDetectResults(std::vector<Object>& vObjects, RoIScaleInfo& roiScaleInfo, cv::Size& imgSize);

	int sendRoiScaleInfo();

private:
	QMutex m_mutex;
	bool m_hasNewFrame;
	cv::Mat m_nextFrame;

	RoIScaleInfo m_currROIScaleInfo;

	ROIDetection* m_roiScaleDetector;

	DataBuffer* m_dataBuffer;
};

