#pragma once

#include <QFile>
#include <QTextStream>
#include <QtXml>
#include <QDomDocument>
#include <QDomElement>
#include <QDebug>
#include <QDateTime>
#include <QDir>
#include <QString>
#include "type_define.h"

class ParamSaver
{
public:
	ParamSaver(QString patientName);
	~ParamSaver();

	void initParamSaver(const QString& stinstID);

	void addData(const ParamData& data);

	bool isPatientNameEmpty()
	{
		return m_filename.isEmpty();
	}

private:
	int xmlFileInitialize(QString& patientName);

	int paramHandling(ParamData& paramData);

private:
	QString m_filename, m_rootDir;
	QString m_currStudyID;
};
