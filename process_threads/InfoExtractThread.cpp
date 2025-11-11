#include "InfoExtractThread.h"

InfoExtractThread::InfoExtractThread(QObject* parent, DataBuffer* dataBuffer)
	: QThread(parent)
	, m_infoExtractor(new InfoExtractor())
	, m_roiDataBuffer(dataBuffer)
{
	qInfo() << "InfoExtractThread initalized.";
}

void InfoExtractThread::run()
{
	while (!isInterruptionRequested())
	{
		cv::Mat frame;
		RoIScaleInfo currROIScaleInfo;

		{
			QMutexLocker locker(&m_mutex);
			if (m_bHasNewFrame)
			{
				frame = m_nextFrame.clone();
				m_bHasNewFrame = false;
			}

			if (m_bHasNewROIScaleInfo)
			{
				currROIScaleInfo.fRadius = m_currROIScaleInfo.fRadius;
				currROIScaleInfo.roiRect = m_currROIScaleInfo.roiRect;
				currROIScaleInfo.specScaleRect = m_currROIScaleInfo.specScaleRect;
				m_bHasNewROIScaleInfo = false;
			}
		}

		if (!frame.empty())
		{
			if (!currROIScaleInfo.specScaleRect.empty())
				m_infoExtractor->doInfoExtract(frame, m_currROIScaleInfo, m_currScaleInfo, m_currPatientInfo, m_modeInfo);
			else
				m_infoExtractor->doInfoExtract(frame, m_currScaleInfo, m_currPatientInfo, m_modeInfo);
			//m_infoExtractor->doInfoExtract(frame, m_currScaleInfo, m_currPatientInfo, m_modeInfo);
			
			QString patientName = StdStringToQString(m_currPatientInfo.patientName);
			QString patientID = StdStringToQString(m_currPatientInfo.patientID);
			emit sigPatientInfo(patientName, patientID);
			//emit sigPatientName(patientName);

			QVariant qTempROIScaleInfo, qTempScaleInfo, qTempModeInfo;
			//qTempROIScaleInfo.setValue(m_currROIScaleInfo);
			qTempScaleInfo.setValue(m_currScaleInfo);
			qTempModeInfo.setValue(m_modeInfo);
			emit sigScaleModeInfoAvailable(qTempScaleInfo, qTempModeInfo);
			//emit sigScaleInfoAvailable(qTempScaleInfo);

			m_modeInfo.clear();
			m_currScaleInfo.clear();
			m_currPatientInfo.clear();
		}
	}
}

void InfoExtractThread::setVideoFrame(const QImage frame)
{
	QMutexLocker locker(&m_mutex);
	cv::Mat cvFrame = GeneralUtils::qImage2cvMat(frame);
	m_nextFrame = cvFrame.clone();
	m_bHasNewFrame = true;
}

int InfoExtractThread::inputVideoFrame(cv::Mat& frame)
{
	QMutexLocker locker(&m_mutex);
	m_nextFrame = frame.clone();
	m_bHasNewFrame = true;
	return 1;
}

void InfoExtractThread::setROIScaleInfo(QVariant qROIScaleInfo)
{
	QMutexLocker locker(&m_mutex);
	m_currROIScaleInfo = qROIScaleInfo.value<RoIScaleInfo>();
	m_bHasNewROIScaleInfo = true;
}

//void InfoExtractThread::sigScaleInfoAvailable(QVariant qScaleInfo)
//{
//}
