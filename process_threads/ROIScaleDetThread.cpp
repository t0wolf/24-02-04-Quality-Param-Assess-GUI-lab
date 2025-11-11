#include "ROIScaleDetThread.h"

ROIScaleDetThread::ROIScaleDetThread(QObject* parent, DataBuffer* roiDataBuffer)
	: QThread(parent)
	, m_roiScaleDetector(new ROIDetection("D:\\Resources\\20240221\\quality_control_models\\roi_detection_models\\roi_detection_0328.engine", cv::Size(640, 640)))
	, m_dataBuffer(roiDataBuffer)
{
	//qInfo() << "ROIScaleDetThread";
	m_currROIScaleInfo = RoIScaleInfo();
	//m_roiScaleDetector = new ROIDetection("D:\\Resources\\20240221\\quality_control_models\\roi_detection_models\\roi_detection.engine", cv::Size(640, 640));
}

void ROIScaleDetThread::run()
{
	while (!isInterruptionRequested())
	{
		cv::Mat frame;

		if (m_hasNewFrame)
		{
			QMutexLocker locker(&m_mutex);
			frame = m_nextFrame.clone();
			m_hasNewFrame = false;
		}

		if (!frame.empty())
		{
			std::vector<Object> vObjects = m_roiScaleDetector->doInference(frame);
			cv::Size frameSize = frame.size();
			//qInfo() << "object size: " << vObjects.size();
			if (!vObjects.empty())
				parseRoiDetectResults(vObjects, m_currROIScaleInfo, frameSize);
			else
				m_currROIScaleInfo.clear();

			sendRoiScaleInfo();
			RoIScaleInfo tempScaleInfo = m_currROIScaleInfo;
			m_dataBuffer->setData(tempScaleInfo);
		}
	}
}

int ROIScaleDetThread::inputVideoFrame(cv::Mat& frame)
{
	//QMutexLocker locker(&m_mutex);
	m_mutex.lock();
	m_nextFrame = frame.clone();
	m_hasNewFrame = true;
	m_mutex.unlock();
	return 1;
}

int ROIScaleDetThread::doROIDetInfer(cv::Mat& currFrame, std::vector<Object>& vObjects)
{
	return 1;
}

int ROIScaleDetThread::parseRoiDetectResults(std::vector<Object>& vObjects, RoIScaleInfo& roiScaleInfo, cv::Size& imgSize)
{
	roiScaleInfo.clear();
	for (auto& object : vObjects)
	{
		if (object.label == 0)
		{
			//qInfo() << "RoI detected.";
			if (object.rect.height >= imgSize.height / 2)
			{
				roiScaleInfo.roiRect = object.rect;
				roiScaleInfo.fRadius = object.rect.height;
			}
		}
		else if (object.label == 1)
		{
			//qInfo() << "Scale detected.";
			roiScaleInfo.specScaleRect = object.rect;
		}
	}

	return 1;
}

int ROIScaleDetThread::sendRoiScaleInfo()
{
	QVariant qVar;
	qVar.setValue(m_currROIScaleInfo);
	emit sigROIScaleAvailable(qVar);

	return 1;
}

void ROIScaleDetThread::setVideoFrame(const QImage frame)
{
	QMutexLocker locker(&m_mutex);
	cv::Mat cvFrame = GeneralUtils::qImage2cvMat(frame);
	m_nextFrame = cvFrame.clone();
	m_hasNewFrame = true;
}