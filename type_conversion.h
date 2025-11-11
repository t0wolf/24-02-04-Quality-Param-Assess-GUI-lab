#pragma once
#include <iostream>
#include <QWidget>
#include <QString>
#include <QVector>
#include <QMap>

// QString to std::string
std::string QStringToStdString(const QString& qstr);

// std::string to QString
QString StdStringToQString(const std::string& str);

template<typename K, typename V>
std::map<K, V> QMapToStdMap(const QMap<K, V>& qmap);

// std::map to QMap
template<typename K, typename V>
QMap<K, V> StdMapToQMap(const std::map<K, V>& stdMap);

// QVector to std::vector
template<typename T>
std::vector<T> QVectorToStdVector(const QVector<T>& qvec);

// std::vector to QVector
template<typename T>
QVector<T> StdVectorToQVector(const std::vector<T>& stdVec);