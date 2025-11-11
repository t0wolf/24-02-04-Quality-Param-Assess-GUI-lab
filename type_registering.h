#pragma once
#include <qmetatype.h>
#include <opencv2/opencv.hpp>
#include <QVector>
#include <type_define.h>

//Q_DECLARE_METATYPE(QVariant*)
Q_DECLARE_METATYPE(cv::Rect)
Q_DECLARE_METATYPE(QVector<float>)
Q_DECLARE_METATYPE(QVector<int>)
Q_DECLARE_METATYPE(QVector<cv::Mat>)
Q_DECLARE_METATYPE(ScaleInfo)
Q_DECLARE_METATYPE(ModeInfo)
Q_DECLARE_METATYPE(RoIScaleInfo)