#pragma once
#include <QString>
#include <QVector>
#include <QMap>

struct Patient {
    QString name;  // 病人姓名
    QString id;    // 病人 ID
    QMap<QString, QVector<double>> measurements; // 存储各种测量参数
};

class PatientManager {
public:
    PatientManager() = default;

    // 添加病人
    void addPatient(const QString& name, const QString& id, const QMap<QString, QVector<double>>& measurements);

    // 获取病人数
    int patientCount() const;

    // 获取病人信息
    const Patient& getPatient(int index) const;

    // 获取所有病人姓名列表
    QStringList getPatientNames() const;

private:
    QVector<Patient> patients; // 病人信息列表
};
