#include "ParamAssessThread.h"

ParamAssessThread::ParamAssessThread(QObject *parent, ConfigParse* config)
	: QThread(parent)
	, m_paramAssesser(new ParamAssesser(config))
{

}

void ParamAssessThread::run()
{
	while (!this->isInterruptionRequested())
	{
		if (m_bIsVideoClipsUpdate)
		{
			qDebug() << "[I] ParamAssessThread::run";
			std::vector<int> tempKeyframeIdx;
			std::string tempViewName;
			std::vector<cv::Mat> tempVideoClip;

			if (m_vVideoClip.empty() && m_keyframeIdx.empty() == 0)
			{
				//emit sigParamsResult(m_currViewName, QVariant(), QVariant());
				m_bIsVideoClipsUpdate = false;
				continue;
			}

			{
				QMutexLocker locker(&m_mutex);
				tempKeyframeIdx = std::vector<int>(m_keyframeIdx.begin(), m_keyframeIdx.end());
				tempViewName = m_currViewName.toStdString();
				tempVideoClip = std::vector<cv::Mat>(m_vVideoClip.begin(), m_vVideoClip.end());
			}

			m_paramAssesser->doInferece(tempViewName, tempVideoClip, tempKeyframeIdx, m_currModeInfo);

			bool bIsFuncBiplaneMode = false;
			QString currViewName = QString::fromStdString(tempViewName);
			QVariant valueVar, premiumsVar;
			if (m_currModeInfo.mode == "B-Mode")
			{
				if (m_currViewName == "PLAX")
				{
					valueVar.setValue(m_paramAssesser->getParamValues());
					premiumsVar.setValue(m_paramAssesser->getParamPremiums());
				}

				else if (m_currViewName == "A4C" || m_currViewName == "A2C")
				{
					valueVar.setValue(m_paramAssesser->getFuncParamValues());
					premiumsVar.setValue(m_paramAssesser->getFuncParamPremiums());
					if (m_paramAssesser->isFuncBiplaneMode())
						currViewName = "BP";
				}
			}
			else if (m_currModeInfo.mode == "Doppler-Mode")
			{
				valueVar.setValue(m_paramAssesser->getSpecParamValues());
				premiumsVar.setValue(m_paramAssesser->getSpecParamPremiums());
				//currViewName = "Spectrum";
			}
					 
			emit sigParamsResult(currViewName, valueVar, premiumsVar);

			m_bIsVideoClipsUpdate = false;
			m_paramAssesser->clearAllParams();
		}
	}

	//std::vector<int> tempKeyframeIdx;
	//std::string tempViewName;
	//std::vector<cv::Mat> tempVideoClip;

	//if (m_vVideoClip.empty() && m_keyframeIdx.empty() == 0)
	//{
	//	//emit sigParamsResult(m_currViewName, QVariant(), QVariant());
	//	m_bIsVideoClipsUpdate = false;
	//	return;
	//	//continue;
	//}

	//QMutexLocker locker(&m_mutex);
	//tempKeyframeIdx = std::vector<int>(m_keyframeIdx.begin(), m_keyframeIdx.end());
	//tempViewName = m_currViewName.toStdString();
	//tempVideoClip = std::vector<cv::Mat>(m_vVideoClip.begin(), m_vVideoClip.end());

	//m_paramAssesser->doInferece(tempViewName, tempVideoClip, tempKeyframeIdx, m_currModeInfo);

	//locker.unlock();

	//QVariant valueVar, premiumsVar;
	//if (m_currModeInfo.mode == "B-Mode")
	//{
	//	if (m_currViewName == "PLAX")
	//	{
	//		valueVar.setValue(m_paramAssesser->getParamValues());
	//		premiumsVar.setValue(m_paramAssesser->getParamPremiums());
	//	}

	//	else if (m_currViewName == "A4C")
	//	{
	//		valueVar.setValue(m_paramAssesser->getFuncParamValues());
	//		premiumsVar.setValue(m_paramAssesser->getFuncParamPremiums());
	//	}
	//}
	//else if (m_currModeInfo.mode == "Doppler-Mode")
	//{
	//	valueVar.setValue(m_paramAssesser->getSpecParamValues());
	//	premiumsVar.setValue(m_paramAssesser->getSpecParamPremiums());
	//}

	//emit sigParamsResult(m_currViewName, valueVar, premiumsVar);

	//m_paramAssesser->clearAllParams();
}

int ParamAssessThread::inputParamAssess(QString& viewName, QVariant& videoClip, QVariant& keyframeIdxes, QVariant& modeInfo)
{
	QMutexLocker locker(&m_mutex);
	qDebug() << "[I] ParamAssessThread::setParamAssessInput";
	m_currViewName = viewName;
	m_vVideoClip = videoClip.value<QVector<cv::Mat>>();
	m_keyframeIdx = keyframeIdxes.value<QVector<int>>();
	m_currModeInfo = modeInfo.value<ModeInfo>();
	m_bIsVideoClipsUpdate = true;

	return 1;
}

void ParamAssessThread::performParamAssess(QString viewName, QVariant videoClip, QVariant keyframeIdxes, QVariant modeInfo)
{
	qDebug() << "[I] ParamAssessThread::setParamAssessInput";
	m_currViewName = viewName;
	m_vVideoClip = videoClip.value<QVector<cv::Mat>>();
	m_keyframeIdx = keyframeIdxes.value<QVector<int>>();
	m_currModeInfo = modeInfo.value<ModeInfo>();

	start();
}

void ParamAssessThread::setParamAssessInput(QString viewName, QVariant videoBuffer, QVariant keyframeIdxes, QVariant modeInfo)
{
    QMutexLocker locker(&m_mutex);
	//qDebug() << "[I] ParamAssessThread::setParamAssessInput";
	m_currViewName = viewName;
	m_vVideoClip = videoBuffer.value<QVector<cv::Mat>>();
	m_keyframeIdx = keyframeIdxes.value<QVector<int>>();
	m_currModeInfo = modeInfo.value<ModeInfo>();
	m_bIsVideoClipsUpdate = true;
}

void ParamAssessThread::setPatientName(QString patientName) 
{
	QMutexLocker locker(&m_mutex);
	m_paramAssesser->setPatientName(patientName);
}

void ParamAssessThread::setScaleInfo(QVariant qScaleInfo)
{
	QMutexLocker locker(&m_mutex);
	m_currScaleInfo = qScaleInfo.value<ScaleInfo>();
	float currentLength = m_currScaleInfo.length;
	float currentScale = m_currScaleInfo.fPixelPerUnit;

	m_paramAssesser->m_funcParamsAssesser->setScaleInfo(currentLength, currentScale);
	m_paramAssesser->m_plaxParamsAssesser->m_pMultiLineAssesser->setScaleInfo(currentLength, currentScale);
	m_paramAssesser->m_plaxParamsAssesser->m_pMultiLineAortaAssesser->setScaleInfo(currentLength, currentScale);
	m_paramAssesser->m_plaxParamsAssesser->m_pAvadAssesser->setScaleInfo(currentLength, currentScale);
}