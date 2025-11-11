#pragma once

#include <QWidget>
#include <QThread>
#include <QMutex>
#include <QVariant>

#include "type_conversion.h"
#include "type_define.h"
#include "general_utils.h"
#include "info_extraction/InfoExtractor.h"
#include "type_registering.h"
#include "DataBuffer.h"
//#include "quality_control/roi_detection.h"
//#include "process_threads/ROIScaleDetThread.h"


class InfoExtractThread : public QThread
{
	Q_OBJECT
public:
	InfoExtractThread(QObject* parent = nullptr, DataBuffer* dataBuffer = nullptr);

	void run() override;

	~InfoExtractThread()
	{
		exitThread();
	}

	void exitThread()
	{
		this->requestInterruption();
		this->quit();
		this->wait();
	}

	int inputVideoFrame(cv::Mat& frame);

public slots:
	void setVideoFrame(const QImage frame);

	void setROIScaleInfo(QVariant qROIScaleInfo);

signals:
	void sigPatientInfo(QString patientName, QString patientID);

	//void sigPatientName(QString patientName);

	//void sigROIScaleInfo(QVariant qROIScaleInfo, QVariant qScaleInfo, QVariant qModeInfo);

	//void sigScaleInfo(QVariant qScaleInfo);

	void sigScaleInfoAvailable(QVariant qScaleInfo);

	void sigScaleModeInfoAvailable(QVariant qScaleInfo, QVariant qModeInfo);

private:
	int doROIDetInfer(cv::Mat& currFrame, std::vector<Object>& vObjects);

	int parseRoiDetectResults(std::vector<Object>& vObjects, RoIScaleInfo& roiScaleInfo);

private:
	InfoExtractor* m_infoExtractor;
	//ROIDetection* m_roiDetector;
	//ROIScaleDetThread m_roiScaleThread;

	QMutex m_mutex;

	bool m_bHasNewFrame;
	bool m_bHasNewROIScaleInfo;

	cv::Mat m_nextFrame;
	ScaleInfo m_currScaleInfo;
	ScaleInfo m_currSpecScaleInfo;
	PatientInfo m_currPatientInfo;
	ModeInfo m_modeInfo;
	RoIScaleInfo m_currROIScaleInfo;

	DataBuffer* m_roiDataBuffer;
};

