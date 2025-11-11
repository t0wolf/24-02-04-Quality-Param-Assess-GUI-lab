#pragma once

#include <QCoreApplication>
#include <QThread>
#include <QNetworkAccessManager>
#include <QNetworkRequest>
#include <QNetworkReply>
#include <QUrl>
#include <QDebug>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonParseError>

class HttpJSONReceiver  : public QThread
{
	Q_OBJECT

public:
	explicit HttpJSONReceiver(const QUrl& url, QObject* parent = nullptr)
		: QThread(parent), m_url(url) {}
	~HttpJSONReceiver();

	void run();

public slots:
	void handleReplyFinished();

signals:
	void sigJsonReceived(const QJsonObject& jsonObject);
	void sigJsonReceived(const QJsonArray& jsonArray);

	void sigCheckIDReceived(const QString& checkID);

private:
	QUrl m_url;
	QString m_checkIDKey = "checkID";
};
