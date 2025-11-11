#include "param_assess_thread.h"

ParamAssessThread::ParamAssessThread(QObject *parent)
    : m_ivsPWAssesser(new IVSAndPWAssess())
{

}

ParamAssessThread::~ParamAssessThread()
{
    exitThread();
}

void ParamAssessThread::exitThread()
{
    this->requestInterruption();
    this->quit();
    this->wait();
}

void ParamAssessThread::run()
{
    // view classification

    //
}
