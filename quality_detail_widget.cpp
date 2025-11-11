#include "quality_detail_widget.h"
#include "ui_quality_detail_widget.h"

// QualityDetailWidget::QualityDetailWidget(QWidget *parent, ClickableLabel* clickableLabel)
//     : QWidget(parent)
//     , ui(new Ui::QualityDetailWidget)
//     , m_clickableLabel(clickableLabel)
// {
//     ui->setupUi(this);
//     connect(m_clickableLabel, &ClickableLabel::clicked, this, &QualityDetailWidget::slotsDisplayDetailed);
// }

// QualityDetailWidget::~QualityDetailWidget()
// {
//     delete ui;
// }

// void QualityDetailWidget::slotsDisplayDetailed()
// {
//     this->show();
// }

QualityDetailWidget::QualityDetailWidget(QWidget *parent, QString viewName)
    : QWidget(parent)
    , ui(new Ui::QualityDetailWidget)
    , m_viewName(viewName)
{
    ui->setupUi(this);
    ui->videoPlayLabel->setScene(&m_scene);
    connect(this, &QualityDetailWidget::deleteAvailable, this, &QualityDetailWidget::clearLabelTable);
    // connect(this, &QualityDetailWidget::deleteAvailable,)
    connect(this, &QualityDetailWidget::clearLabelAvailable, this, &QualityDetailWidget::setLabelTableClear);
}

QualityDetailWidget::~QualityDetailWidget()
{
    delete ui;
}

void QualityDetailWidget::setSavedVideoFrame(QImage qImage)
{
    //QPixmap qPixmap = QPixmap::fromImage(qImage);
    //ui->videoPlayLabel->setPixmap(qPixmap);
    QPixmap tempPixmap = QPixmap::fromImage(qImage);
    m_scene.clear();
    m_scene.addPixmap(tempPixmap);
    adaptToView();
}


void QualityDetailWidget::on_pushButton_clicked()
{
    emit deleteAvailable(m_viewName);
    emit clearLabelAvailable();
}

void QualityDetailWidget::clearLabelTable()
{
    //ui->videoPlayLabel->setPixmap(QPixmap());
    m_scene.clear();
    m_scene.addPixmap(QPixmap());
    ui->qualityScoresTable->clear();
}

void QualityDetailWidget::setLabelTableClear()
{
    clearLabelTable();
}

void QualityDetailWidget::setQualityRules(rapidjson::Value& qualityRules)
{
    m_qualityRules = qualityRules;
}

float QualityDetailWidget::setQualityScores(QVariant currResult)
{
    auto qVGrades = currResult.value<QVector<float>>();
    int counter = 0;
    float fTotalScore = 0.0f;

    for (auto& rule : m_qualityRules["rules"].GetArray())
    {
        QTableWidgetItem* ruleItem = new QTableWidgetItem();
        ruleItem->setText(rule.GetString());
        ui->qualityScoresTable->setItem(counter, 0, ruleItem);

        float fCurrScore = qVGrades[counter];
        fTotalScore += fCurrScore;

        QTableWidgetItem* scoreItem = new QTableWidgetItem();
        bool bIsPassed = fCurrScore > 0.0f;
        scoreItem->setText(QString::number(qVGrades[counter]));

        if (bIsPassed)
        {
            QColor color("#00FF00");
            scoreItem->setForeground(QBrush(color));
            QFont font = scoreItem->font();
            font.setBold(true);
            scoreItem->setFont(font);
        }
        ui->qualityScoresTable->setItem(counter, 1, scoreItem);

        counter++;
    }

    return fTotalScore;
}

void QualityDetailWidget::adaptToView()
{

    QRectF rectItem = m_scene.itemsBoundingRect();
    QRectF rectView = ui->videoPlayLabel->rect();
    qreal ratioView = rectView.height() / rectView.width();
    qreal ratioItem = rectItem.height() / rectItem.width();
    if (ratioView > ratioItem)
    {
        rectItem.moveTop(rectItem.width() * ratioView - rectItem.height());
        rectItem.setHeight(rectItem.width() * ratioView);

        rectItem.setWidth(rectItem.width() * 1.0);
        rectItem.setHeight(rectItem.height() * 1.0);
    }
    else
    {
        rectItem.moveLeft(rectItem.height() / ratioView - rectItem.width());
        rectItem.setWidth(rectItem.height() / ratioView);

        rectItem.setWidth(rectItem.width() * 1.0);
        rectItem.setHeight(rectItem.height() * 1.0);
    }
    ui->videoPlayLabel->fitInView(rectItem, Qt::KeepAspectRatio);

}

