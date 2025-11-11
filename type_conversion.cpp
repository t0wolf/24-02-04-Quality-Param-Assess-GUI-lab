#include "type_conversion.h"

// QString to std::string
std::string QStringToStdString(const QString& qstr) {
    return qstr.toStdString();
}

// std::string to QString
QString StdStringToQString(const std::string& str) {
    return QString::fromStdString(str);
}

template<typename K, typename V>
std::map<K, V> QMapToStdMap(const QMap<K, V>& qmap) {
    std::map<K, V> stdMap;
    for (auto iter = qmap.begin(); iter != qmap.end(); ++iter) {
        stdMap.insert({ iter.key(), iter.value() });
    }
    return stdMap;
}

// std::map to QMap
template<typename K, typename V>
QMap<K, V> StdMapToQMap(const std::map<K, V>& stdMap) {
    QMap<K, V> qmap;
    for (const auto& pair : stdMap) {
        qmap.insert(pair.first, pair.second);
    }
    return qmap;
}

// QVector to std::vector
template<typename T>
std::vector<T> QVectorToStdVector(const QVector<T>& qvec) {
    std::vector<T> stdVec(qvec.begin(), qvec.end());
    return stdVec;
}

// std::vector to QVector
template<typename T>
QVector<T> StdVectorToQVector(const std::vector<T>& stdVec) {
    return QVector<T>(stdVec.begin(), stdVec.end());
}