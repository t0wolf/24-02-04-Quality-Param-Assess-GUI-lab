#include "param_assess_widget.h"
#include "ui_param_assess_widget.h"

ParamAssessWidget::ParamAssessWidget(ProgressSuperThread* progressSuperThread, QWidget *parent)
    : QWidget(parent)
    , ui(new Ui::ParamAssessWidget)
    , m_progressSuperThread(progressSuperThread)
{
    ui->setupUi(this);
    connect(this, &ParamAssessWidget::paramNameAvailable, m_progressSuperThread, &ProgressSuperThread::setProgressMapUpdate);
}

ParamAssessWidget::~ParamAssessWidget()
{
    delete ui;
}

void ParamAssessWidget::setStructureParamEvents(QVariant paramValues)
{
    int row = 0;
    QMap<QString, float> tempParamValues = paramValues.value<QMap<QString, float>>();
    for (auto it = tempParamValues.begin(); it != tempParamValues.end(); it++)
    {
        QTableWidgetItem * keyItem = new QTableWidgetItem(it.key());
        ui->paramTable->setItem(row, 0, keyItem);

        QTableWidgetItem * valueItem = new QTableWidgetItem(QString::number(it.value()));
        ui->paramTable->setItem(row, 1, valueItem);

        ++row;
        emit(paramNameAvailable(it.key()));
    }
}

// void ParamAssessWidget::setSpectrumParamEvents(QMap<QString, float> paramValues)
// {
//     int row = 0;
//     for (auto it = paramValues.begin(); it != paramValues.end(); it++)
//     {
//         QTableWidgetItem * keyItem = new QTableWidgetItem(it.key());
//         ui->spectrumParamTable->setItem(row, 0, keyItem);

//         QTableWidgetItem * valueItem = new QTableWidgetItem(QString::number(it.value()));
//         ui->spectrumParamTable->setItem(row, 1, valueItem);

//         ++row;
//     }
// }

// void ParamAssessWidget::setFuncParamEvents(QMap<QString, float> paramValues)
// {
//     int row = 0;
//     for (auto it = paramValues.begin(); it != paramValues.end(); it++)
//     {
//         QTableWidgetItem * keyItem = new QTableWidgetItem(it.key());
//         ui->functionParamTable->setItem(row, 0, keyItem);

//         QTableWidgetItem * valueItem = new QTableWidgetItem(QString::number(it.value()));
//         ui->functionParamTable->setItem(row, 1, valueItem);

//         ++row;
//     }
// }
