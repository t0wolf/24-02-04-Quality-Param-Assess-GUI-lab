#ifndef PARAMASSESSTHREAD_H
#define PARAMASSESSTHREAD_H

#include <iostream>
#include <QThread>
#include <opencv2/opencv.hpp>
#include "param_assessment/ivs_and_pw_assess.h"

class ParamAssessThread : QThread
{
public:
    ParamAssessThread(QObject *parent = nullptr);

    ~ParamAssessThread();

    void exitThread();

    void run() override;

private:
    IVSAndPWAssess* m_ivsPWAssesser;
};

#endif // PARAMASSESSTHREAD_H
