#include "PatientManager.h"

// 添加病人
void PatientManager::addPatient(const QString& name, const QString& id, const QMap<QString, QVector<double>>& measurements) {
    Patient newPatient;
    newPatient.name = name;
    newPatient.id = id;
    newPatient.measurements = measurements;

    patients.append(newPatient);
}

// 获取病人数
int PatientManager::patientCount() const {
    return patients.size();
}

// 获取病人信息
const Patient& PatientManager::getPatient(int index) const {
    if (index >= 0 && index < patients.size()) {
        return patients[index];
    }
    
}

// 获取所有病人姓名列表
QStringList PatientManager::getPatientNames() const {
    QStringList names;
    for (const Patient& patient : patients) {
        names.append(patient.name + " (ID: " + patient.id + ")");
    }
    return names;
}