#include "settingsdialog.h"
#include "ui_settingsdialog.h"

SettingsDialog::SettingsDialog(QWidget *parent, ConfigParse *config)
    : QDialog(parent)
    , m_config(config)
    , ui(new Ui::SettingsDialog)
{
    ui->setupUi(this);
    m_originalLines.clear();
    m_keyToLineIndex.clear();
    m_originalValuesMap.clear();
    initconfigsWidget();
}

SettingsDialog::~SettingsDialog()
{
    delete ui;
}

void SettingsDialog::initconfigsWidget()
{
    ui->configsWidget->setRowCount(0);  // 清空行

    //// 打开 YAML 配置文件
    //QFile file("D:\\Resources\\20240221\\configs.yaml");
    //if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
    //    qDebug() << "无法打开 configs.yaml";
    //    return;
    //}

    //// 读取文件内容
    //QByteArray fileData = file.readAll();
    //file.close();

    // 解析 YAML
    //YAML::Node config = YAML::Load(fileData.constData());
    YAML::Node config = m_config->getAllNode();

    if (!config.IsMap()) {
        qDebug() << "YAML 文件格式不正确";
        return;
    }

    int displayRow = 0;

    // 遍历 YAML 键值对
    for (YAML::const_iterator it = config.begin(); it != config.end(); ++it) {
        QString key = QString::fromStdString(it->first.as<std::string>());
        QString value = QString::fromStdString(it->second.as<std::string>());

        // 根据字典修改 key
        //int index = m_chineseMap.indexOf(key);
        //if (index != -1 && index < m_englishMap.size()) {
        //    key = m_englishMap[index];
        //}

        if (key.isEmpty()) {
            continue; // 跳过空的 key
        }

        // 插入表格
        ui->configsWidget->insertRow(displayRow);
        ui->configsWidget->setItem(displayRow, 0, new QTableWidgetItem(key));
        ui->configsWidget->setItem(displayRow, 1, new QTableWidgetItem(value));

        // 设置参数名称不可修改
        ui->configsWidget->item(displayRow, 0)->setFlags(Qt::ItemIsSelectable | Qt::ItemIsEnabled);

        // 建立 key 对应信息
        m_keyToLineIndex[key] = displayRow;  // 更新为 displayRow
        m_originalValuesMap[key] = value;

        ++displayRow;
    }

    ui->configsWidget->resizeColumnsToContents();
    //ui->configsWidget->setRowCount(0);  // 清空行

    //QFile file("/home/jetson/resources/configs.txt");
    //if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
    //    qDebug() << "无法打开 configs.txt";
    //    return;
    //}

    //QTextStream in(&file);
    //int displayRow = 0;
    //int lineIndex = 0;

    //while (!in.atEnd()) {
    //    QString rawLine = in.readLine();
    //    m_originalLines.append(rawLine);  // 每一行都保存下来

    //    QString line = rawLine.trimmed();

    //    // 跳过空行和注释行（行首或中间带 # 的）
    //    if (line.isEmpty() || line.startsWith("#")) {
    //        ++lineIndex;
    //        continue;
    //    }

    //    int colonIndex = line.indexOf(":");
    //    if (colonIndex == -1) {
    //        ++lineIndex;
    //        continue; // 跳过无冒号的行
    //    }

    //    // 只取第一个冒号前后内容
    //    QString key = line.left(colonIndex).trimmed();
    //    // 根据字典修改key
    //    int index = m_chineseMap.indexOf(key);
    //    if (index != -1 && index < m_englishMap.size()) {
    //        key = m_englishMap[index];
    //    }
    //    QString value = line.mid(colonIndex + 1).trimmed();

    //    if (key.isEmpty()) {
    //        ++lineIndex;
    //        continue;
    //    }

    //    // 插入表格
    //    ui->configsWidget->insertRow(displayRow);
    //    ui->configsWidget->setItem(displayRow, 0, new QTableWidgetItem(key));
    //    ui->configsWidget->setItem(displayRow, 1, new QTableWidgetItem(value));

    //    // 设置参数名称不可修改
    //    ui->configsWidget->item(displayRow, 0)->setFlags(Qt::ItemIsSelectable | Qt::ItemIsEnabled);

    //    // 设置 key 单元格为不可编辑
    //    ui->configsWidget->item(displayRow, 0)->setFlags(Qt::ItemIsSelectable | Qt::ItemIsEnabled);

    //    // 建立 key 对应信息
    //    m_keyToLineIndex[key] = lineIndex;
    //    m_originalValuesMap[key] = value;

    //    ++displayRow;
    //    ++lineIndex;
    //}

    //file.close();
    //ui->configsWidget->resizeColumnsToContents();
}

//void SettingsDialog::on_okButton_clicked()
//{
//
//    int rowCount = ui->configsWidget->rowCount();
//
//    for (int row = 0; row < rowCount; ++row) {
//        QTableWidgetItem* keyItem = ui->configsWidget->item(row, 0);
//        QTableWidgetItem* valueItem = ui->configsWidget->item(row, 1);
//        if (!keyItem || !valueItem) continue;
//
//        QString key = keyItem->text().trimmed();
//        QString newValue = valueItem->text().trimmed();
//
//        // 原始值
//        if (!m_originalValuesMap.contains(key)) continue;
//        QString oldValue = m_originalValuesMap[key];
//
//        // 如果没有改动，跳过
//        if (newValue == oldValue) continue;
//
//        // 找到原始行号
//        if (!m_keyToLineIndex.contains(key)) continue;
//        int lineIndex = m_keyToLineIndex[key];
//        QString oldLine = m_originalLines[lineIndex];
//
//        int colonIndex = oldLine.indexOf(":");
//        if (colonIndex != -1) {
//            QString newLine = oldLine.left(colonIndex + 1) + " " + newValue;
//            m_originalLines[lineIndex] = newLine;
//        }
//    }
//
//    // 写回原始文件（保留所有未动的行）
//    QFile file("/home/jetson/resources/configs.txt");
//    if (!file.open(QIODevice::WriteOnly | QIODevice::Text | QIODevice::Truncate)) {
//        qDebug() << "无法打开 configs.txt 写入";
//        return;
//    }
//
//    QTextStream out(&file);
//    for (const QString& line : m_originalLines) {
//        out << line << "\n";
//    }
//
//    file.close();
//    this->close();
//}
void SettingsDialog::on_okButton_clicked()
{
    // 读取配置项数
    int rowCount = ui->configsWidget->rowCount();

    // 创建一个 YAML 节点以保存新的配置
    YAML::Node newConfig = m_config->getAllNode();

    for (int row = 0; row < rowCount; ++row) {
        QTableWidgetItem* keyItem = ui->configsWidget->item(row, 0);
        QTableWidgetItem* valueItem = ui->configsWidget->item(row, 1);
        if (!keyItem || !valueItem) continue;

        QString key = keyItem->text().trimmed();
        QString newValue = valueItem->text().trimmed();

        // 原始值
        if (!m_originalValuesMap.contains(key)) continue;
        QString oldValue = m_originalValuesMap[key];

        // 如果没有改动，跳过
        if (newValue == oldValue) continue;

        // 将新值存储在 YAML 节点中
        newConfig[key.toStdString()] = newValue.toStdString();
    }

    // 写入 YAML 文件
    QFile file("D:\\Resources\\20240221\\configs.yaml");
    if (!file.open(QIODevice::WriteOnly | QIODevice::Text | QIODevice::Truncate)) {
        qDebug() << "无法打开 configs.yaml 写入";
        return;
    }

    // 使用 YAML 库将配置写入文件
    file.write(YAML::Dump(newConfig).c_str());
    file.close();

    // 关闭对话框
    this->close();
}

void SettingsDialog::on_cancleButton_clicked()
{
    this->close();
}
