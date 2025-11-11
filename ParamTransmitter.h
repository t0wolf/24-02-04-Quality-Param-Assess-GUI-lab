#pragma once

#include <QObject>
#include <QString>
#include <QCoreApplication>
#include <QDomDocument>
#include <QNetworkAccessManager>
#include <QNetworkReply>
#include <QNetworkRequest>
#include <QHttpMultiPart>
#include <QByteArray>
#include <QUrl>
#include <QUrlQuery>
#include <QDebug>
#include <QImage>
#include <QBuffer>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonArray>
#include "PatientManagerWindow.h"
#include "type_define.h"
#include "config_parse.h"
#include "QtLogger.h"


class ParamTransmitter : public QObject
{
	Q_OBJECT

public:
	ParamTransmitter(ConfigParse* configParser, QObject *parent);

    ParamTransmitter(ConfigParse* configParser, PatientManagerWindow* patientManageWindow, QObject* parent);

	~ParamTransmitter();

	void initParamTransmitter(const QString& stinstID);

    void initParamTransmitter(const QString& stinstID, const QString& patientName, const QString& patientID);

    int addDataAndSendToServer(const ParamData& paramData);

    int addDataToDoc(const ParamData& data);

    int addDataToJson(const ParamData& data);

    int sendDataToServer(const QString& strViewName, const QImage& premiumImage);

    int sendDataToServer(const ParamData& paramData);

    int sendDataToServer();

    ParamData handleFuncParamsNames(const ParamData& paramData);

    int computeEDivedeA();

    int clearParamJsonArray();

private slots:
    void handleNetworkReply();

public slots:
    void slotHandleQualityScore(QString qualityViewName, QString qualityScore);

private:
    QNetworkAccessManager* m_netManager;
    QString m_workStationIP;
    QString m_workStationPort;

    QMutex m_mutex;
    QString m_currQualityScore, m_currQualityViewName;
    QString m_workStationUrl;
    QString m_currStudyInstID, m_currPatientName, m_currPatientID;
    QDomDocument m_currXmlData;
    QDomElement m_currRoot;
    QJsonDocument m_currJsonData;
    QJsonArray m_currJsonArray;
    PatientManagerWindow* m_patientManageWindow;

private:
    inline QString base64Encode(const QString& data) 
    {
        return data.toUtf8().toBase64();
    }

    void initDocument();

    QByteArray imageToBase64(const QString& imagePath);

    QByteArray imageToBase64(const QImage& image);

};
