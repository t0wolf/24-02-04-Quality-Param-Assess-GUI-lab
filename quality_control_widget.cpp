#include "quality_control_widget.h"
#include "ui_quality_control_widget.h"

QualityControlWidget::QualityControlWidget(QWidget *parent)
    : QWidget(parent)
    , ui(new Ui::QualityControlWidget)
    // , m_modelInferenceThread(new ModelsInferenceThread())
{
    ui->setupUi(this);
    parseJSONFile("D:\\Resources\\20240221\\quality_control_rules.json");
    // connect(m_modelInferenceThread, &ModelsInferenceThread::viewNameProcessed, this, &QualityControlWidget::setCurrentViewName);
}

QualityControlWidget::~QualityControlWidget()
{
    delete ui;
}

int QualityControlWidget::parseJSONFile(std::string jsonPath)
{
    std::ifstream ifs(jsonPath);
    if (!ifs.is_open())
    {
        return 0;
    }

    std::stringstream ss;
    ss << ifs.rdbuf();
    ifs.close();

    if (m_viewRulesTotal.Parse(ss.str().c_str()).HasParseError())
    {
        return 0;
    }

    return 1;
}

void QualityControlWidget::setCurrentViewName(const QString viewName)
{
    m_viewName = viewName;
    ui->infoLabel->setText(m_viewName);

    // const rapidjson::Value& qualityRules = m_viewRulesTotal[viewName.toStdString().c_str()].GetObject();
    // const rapidjson::Value& qualityRules = m_viewRulesTotal["PLAX"].GetObj();

    // int counter = 0;
    // for (auto& rule : qualityRules["rules"].GetArray())
    // {
    //     QTableWidgetItem* newItem = new QTableWidgetItem();
    //     newItem->setText(rule.GetString());
    //     ui->qualityRulesTable->setItem(counter, 0, newItem);
    //     counter++;
    // }
}

// int QualityControlWidget::parseQualityScores(std::string& viewName, singleResult& videoResult)
// {

// }

void QualityControlWidget::setCurrentViewQualityScores(QVariant currResult)
{
    auto qVGrades = currResult.value<QVector<float>>();
    const rapidjson::Value& qualityRules = m_viewRulesTotal["PLAX"].GetObj();
    int counter = 0;
    for (auto& rule : qualityRules["rules"].GetArray())
    {
        QTableWidgetItem* ruleItem = new QTableWidgetItem();
        ruleItem->setText(rule.GetString());
        ui->qualityRulesTable->setItem(counter, 0, ruleItem);

        QTableWidgetItem* scoreItem = new QTableWidgetItem();
        scoreItem->setText(QString::number(qVGrades[counter]));
        ui->qualityRulesTable->setItem(counter, 1, scoreItem);

        counter++;
    }
}
