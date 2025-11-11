#pragma once

#include <QObject>
#include <QFile>
#include <QString>
#include <QTextStream>
#include <QImage>
#include <QList>
#include <QMap>
#include <QImage>
#include <QDateTime>
#include <QDir>
#include <QMutex>
#include <opencv2/opencv.hpp>
#include "general_utils.h"


class ParamAssessHTMLShower : public QObject
{
	Q_OBJECT

public:
	ParamAssessHTMLShower(QObject *parent, QString strHTMLSavePath);
	~ParamAssessHTMLShower();

	int updateParamValues(QString strViewName, QMap<QString, QString>& mapParamValues, QMap<QString, QImage>& mapParamPremiums);

public slots:
	void slotReceiveParamValuesPremiums(QString viewName, QVariant paramValues, QVariant paramPremium);

	void slotReceiveStudyInstID(const QString& strStudyInstID);

	void slotReceivePatientName(const QString& strPatientName);

private:
	QString loadHtmlTemplate();

	QString createBaseHtml();

	QString loadExistHtmlFile();

	int saveHtmlFile(QString& strHtmlContent);

	QString savePremiumImageToDisk(QString& strEventName, QImage& img);

	QString saveSpecInferenceImage(QString& eventName, cv::Mat& img, bool isPremium);

	int saveStructInferenceImage(QString viewName,
		std::vector<cv::Mat>& vInferVideoClip,
		std::vector<int>& vKeyframeIndexes,
		QMap<QString, QImage>& qmPremiums);

private:
	QMutex m_mutex;
	QString m_strHTMLFilePath, m_strRootPath, m_strImageSaveRootPath;
	QString m_strCurrInstID, m_strPrevInstID;
	QString m_strCurrPatientName, m_strPrevPatientName;

	QString m_specImageSaveRootPath = "D:/Data/Saved-Images/Doppler-Mode";

	QString m_structImageSaveRootPath = "D:/Data/Saved-Images/B-Mode";

	QDateTime m_lastSpecSaveTime, m_lastStructSaveTime;
};
