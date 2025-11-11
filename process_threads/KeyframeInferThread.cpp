#include "KeyframeInferThread.h"

KeyframeInferThread::KeyframeInferThread(QObject* parent, KeyframeDataBuffer* keyDataBuffer)
	: QThread(parent)
	, m_keyDataBuffer(keyDataBuffer)
	, m_keyframeDetector(new KeyframeDetector())
{}

void KeyframeInferThread::run()
{
	while (!isInterruptionRequested())
	{
		if (m_keyDataBuffer->hasNewKeyDetInfo())
		{
			//QPair<QString, QString> pairKeyInfo = qMakePair(QString(""), QString(""));
			KeyframeData currKeyframeData = m_keyDataBuffer->getKeyDetInfo();

			cv::Mat currFrame = currKeyframeData.inferFrame;
			QString strViewName = currKeyframeData.strViewName;
			QString strInferMode = currKeyframeData.strInferMode;

			if (currFrame.empty() || strViewName.isEmpty() || strInferMode.isEmpty())
			{
				clearAllMemoryCache();
				continue;
			}

			std::vector<PeakInfo> vecPeakInfos;
			qDebug() << QString("[I] Ready to Infer %1 Module...").arg(strInferMode);
			m_vImagesBuffer.push_back(currFrame.clone());

			m_keyframeDetector->doInference(currFrame, strViewName.toStdString(), strInferMode.toStdString(), vecPeakInfos);
			if (strInferMode == "sgta")
			{
				//vecPeakInfos = std::vector<PeakInfo>(vecPeakInfos.begin(), vecPeakInfos.begin() + vecPeakInfos.size() / 2);

				//if (vecPeakInfos.size() >= 2)  // 能够匹配到一个ED和一个ES
				//{
				//	clearAllMemoryCache();
				//}
				//else  // 如果匹配不到一个ED和一个ES，则不清空memory，连着下一次接着检测关键帧
				//{
				//	vecPeakInfos = std::vector<PeakInfo>{ PeakInfo{ -10000, -1, -1, -1 } };
				//}

				if (vecPeakInfos.empty())
				{
					PeakInfo tempPeakInfo{ -10000, -10000, -10000, -10000 };
					vecPeakInfos = { tempPeakInfo };
				}

				QtLogger::instance().logMessage(QString("[I] LGTA Module Inference Done."));
				saveInferenceImage(strViewName, vecPeakInfos, m_vImagesBuffer);
				m_keyDataBuffer->addPeakResult(vecPeakInfos);

				QtLogger::instance().logMessage(QString("[I] PeakInfo Size: %1.").arg(vecPeakInfos.size()));

				clearAllMemoryCache();
				m_keyDataBuffer->clearKeyframeQueue();
			}
		}
	}
}

int KeyframeInferThread::clearAllMemoryCache()
{
	m_keyframeDetector->clearFeatMemory();
	m_vImagesBuffer.clear();
	return 1;
}

int KeyframeInferThread::saveInferenceImage(QString& viewName, std::vector<PeakInfo>& vecKeyframePeakInfo, std::vector<cv::Mat>& img)
{
	if (img.empty())
		return 0;

	QString subFolderName = generateDate();
	QString folderPath = m_videoSaveRootPath + "/" + viewName;
	QDir().mkpath(folderPath);
	
	QString saveFolderPath = folderPath + "/" + subFolderName + "_cycle";
	QDir().mkpath(saveFolderPath);
	
	for (int i = 0; i < img.size(); ++i)
	{
		bool bIsKeyframe = false, bIsED = false, bIsES = false;
		for (auto& peak : vecKeyframePeakInfo)
		{
			if (i == std::abs(peak.index))
			{
				bIsKeyframe = true;
				if (peak.index < 0)
					bIsES = true;
				else
					bIsED = true;

				break;
			}
		}
		QString filePath;
		if (bIsKeyframe)
		{
			if (bIsED)
				filePath = saveFolderPath + "/" + QString("image_%1_ED.jpg").arg(i);
			else if (bIsES)
				filePath = saveFolderPath + "/" + QString("image_%1_ES.jpg").arg(i);
		}
		else
			filePath = saveFolderPath + "/" + QString("image_%1.jpg").arg(i);
			
		cv::imwrite(filePath.toStdString(), img[i]);
	}
	
	return 1;
}
