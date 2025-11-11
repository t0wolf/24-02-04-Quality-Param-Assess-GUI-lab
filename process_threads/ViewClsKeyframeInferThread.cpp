#include "ViewClsKeyframeInferThread.h"

ViewClsKeyframeInferThread::ViewClsKeyframeInferThread(QObject *parent, ViewClsInferBuffer* clsDataBuffer, ConfigParse* config)
	: QThread(parent)
	, m_clsDataBuffer(clsDataBuffer)
{
	// 从配置文件中读取 LVEF 模型路径
	std::string strViewclsBackbonePath, strViewclsSwinheadPath;
	if (config->getSpecifiedNode("VIEWCLS_BACKBONE_PATH", strViewclsBackbonePath)) {
		if (!GeneralUtils::fileExists(strViewclsBackbonePath)) {
			QtLogger::instance().logMessage(QString::fromStdString("[E] Viewcls Backbone path does not exist: ") +
				QString::fromStdString(strViewclsBackbonePath));
		}
	}

	if (config->getSpecifiedNode("VIEWCLS_SWINHEAD_PATH", strViewclsSwinheadPath)) {
		if (!GeneralUtils::fileExists(strViewclsSwinheadPath)) {
			QtLogger::instance().logMessage(QString::fromStdString("[E] Viewcls Swinhead path does not exist: ") +
				QString::fromStdString(strViewclsSwinheadPath));
		}
	}

	if (!strViewclsBackbonePath.empty() || !strViewclsSwinheadPath.empty())
	{
		m_viewClsInferer = new ViewClsInferer(strViewclsBackbonePath, strViewclsSwinheadPath);

		QtLogger::instance().logMessage(QString::fromStdString("[I] Viewcls Model Loaded"));
	}
	else {
		QtLogger::instance().logMessage(QString::fromStdString("[E] Failed to load Viewcls path from config"));
	}
}

void ViewClsKeyframeInferThread::run()
{
	while (!isInterruptionRequested())
	{
		if (m_clsDataBuffer->newImageAvailable())
		{
			cv::Mat frame = m_clsDataBuffer->getViewClsImage();
			if (!frame.empty())
			{
				float fQualityScore = 0.0f;
				bool bIsSwitch = false;
				int viewIdx = m_viewClsInferer->doInference(frame, fQualityScore, bIsSwitch);
				m_vImagesBuffer.push_back(frame);
				parseViewClassInferResult(viewIdx, fQualityScore, bIsSwitch);
			}
			else
			{
				m_clsDataBuffer->clearResultBuffer();
				m_viewClsInferer->clearFeatMemory();
			}
		}
	}
}

int ViewClsKeyframeInferThread::getViewClassClipSize()
{
	return m_viewClsInferer->getClipSize();
}

int ViewClsKeyframeInferThread::parseViewClassInferResult(int viewResultIdx)
{
	if (viewResultIdx >= 0)
	{
		QString qViewName = m_viewNameMapping[viewResultIdx];
		m_clsDataBuffer->addViewClsResult(ViewClsResult{ qViewName, 0.0f });
		qDebug() << "[I] View name: " << qViewName;
		
		m_vImagesBuffer.clear();
		return 1;
	}

	return 0;
}

int ViewClsKeyframeInferThread::parseViewClassInferResult(int viewResultIdx, float fQualityScore)
{
	if (viewResultIdx >= 0)
	{
		QString qViewName = m_viewNameMapping[viewResultIdx];
		m_clsDataBuffer->addViewClsResult(ViewClsResult{ qViewName, fQualityScore, false });
		//qDebug() << "[I] View name: " << qViewName;

		m_vImagesBuffer.clear();
		return 1;
	}

	return 0;
}

int ViewClsKeyframeInferThread::parseViewClassInferResult(int viewResultIdx, float fQualityScore, bool bIsSwitch)
{
	if (viewResultIdx >= 0)
	{
		QString qViewName = m_viewNameMapping[viewResultIdx];
		m_clsDataBuffer->addViewClsResult(ViewClsResult{ qViewName, fQualityScore, bIsSwitch });
		//qDebug() << "[I] View name: " << qViewName;

		m_vImagesBuffer.clear();
		return 1;
	}
	return 0;
}


//int ViewClsKeyframeInferThread::saveInferenceImage(QString& viewName, std::vector<cv::Mat>& vImgs)
//{
//	QString subFolderName = generateDate();
//	QString folderPath = m_videoSaveRootPath + "/" + viewName;
//	QDir().mkpath(folderPath);
//
//	QString saveFolderPath = folderPath + "/" + subFolderName;
//	QDir().mkpath(saveFolderPath);
//
//	for (int i = 0; i < vImgs.size(); ++i)
//	{
//		QString filePath = saveFolderPath + "/" + QString("image_%1.jpg").arg(i);
//		cv::imwrite(filePath.toStdString(), vImgs[i]);
//	}
//
//	return 1;
//}

