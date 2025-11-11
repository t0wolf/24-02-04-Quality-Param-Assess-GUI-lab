/****************************************************************************
** Meta object code from reading C++ file 'quality_display_widget.h'
**
** Created by: The Qt Meta Object Compiler version 67 (Qt 5.15.2)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include <memory>
#include "../../../quality_display_widget.h"
#include <QtCore/qbytearray.h>
#include <QtCore/qmetatype.h>
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'quality_display_widget.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 67
#error "This file was generated using the moc from 5.15.2. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
QT_WARNING_PUSH
QT_WARNING_DISABLE_DEPRECATED
struct qt_meta_stringdata_QualityDisplayWidget_t {
    QByteArrayData data[16];
    char stringdata0[250];
};
#define QT_MOC_LITERAL(idx, ofs, len) \
    Q_STATIC_BYTE_ARRAY_DATA_HEADER_INITIALIZER_WITH_OFFSET(len, \
    qptrdiff(offsetof(qt_meta_stringdata_QualityDisplayWidget_t, stringdata0) + ofs \
        - idx * sizeof(QByteArrayData)) \
    )
static const qt_meta_stringdata_QualityDisplayWidget_t qt_meta_stringdata_QualityDisplayWidget = {
    {
QT_MOC_LITERAL(0, 0, 20), // "QualityDisplayWidget"
QT_MOC_LITERAL(1, 21, 19), // "exitThreadAvailable"
QT_MOC_LITERAL(2, 41, 0), // ""
QT_MOC_LITERAL(3, 42, 23), // "labelInitalizeAvailable"
QT_MOC_LITERAL(4, 66, 22), // "qualityScoresAvailable"
QT_MOC_LITERAL(5, 89, 9), // "qVResults"
QT_MOC_LITERAL(6, 99, 18), // "setCurrentViewName"
QT_MOC_LITERAL(7, 118, 8), // "viewName"
QT_MOC_LITERAL(8, 127, 23), // "setCurrentViewNameImage"
QT_MOC_LITERAL(9, 151, 6), // "qImage"
QT_MOC_LITERAL(10, 158, 23), // "setCurrentViewNameVideo"
QT_MOC_LITERAL(11, 182, 4), // "qVar"
QT_MOC_LITERAL(12, 187, 18), // "setLabelInitialize"
QT_MOC_LITERAL(13, 206, 17), // "setViewUncomplete"
QT_MOC_LITERAL(14, 224, 16), // "setQualityScores"
QT_MOC_LITERAL(15, 241, 8) // "qVResult"

    },
    "QualityDisplayWidget\0exitThreadAvailable\0"
    "\0labelInitalizeAvailable\0"
    "qualityScoresAvailable\0qVResults\0"
    "setCurrentViewName\0viewName\0"
    "setCurrentViewNameImage\0qImage\0"
    "setCurrentViewNameVideo\0qVar\0"
    "setLabelInitialize\0setViewUncomplete\0"
    "setQualityScores\0qVResult"
};
#undef QT_MOC_LITERAL

static const uint qt_meta_data_QualityDisplayWidget[] = {

 // content:
       8,       // revision
       0,       // classname
       0,    0, // classinfo
       9,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       3,       // signalCount

 // signals: name, argc, parameters, tag, flags
       1,    0,   59,    2, 0x06 /* Public */,
       3,    0,   60,    2, 0x06 /* Public */,
       4,    1,   61,    2, 0x06 /* Public */,

 // slots: name, argc, parameters, tag, flags
       6,    1,   64,    2, 0x0a /* Public */,
       8,    2,   67,    2, 0x0a /* Public */,
      10,    2,   72,    2, 0x0a /* Public */,
      12,    1,   77,    2, 0x0a /* Public */,
      13,    1,   80,    2, 0x0a /* Public */,
      14,    2,   83,    2, 0x0a /* Public */,

 // signals: parameters
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void, QMetaType::QVariant,    5,

 // slots: parameters
    QMetaType::Void, QMetaType::QString,    7,
    QMetaType::Void, QMetaType::QString, QMetaType::QImage,    7,    9,
    QMetaType::Void, QMetaType::QString, QMetaType::QVariant,    7,   11,
    QMetaType::Void, QMetaType::QString,    7,
    QMetaType::Void, QMetaType::QString,    7,
    QMetaType::Void, QMetaType::QString, QMetaType::QVariant,    7,   15,

       0        // eod
};

void QualityDisplayWidget::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        auto *_t = static_cast<QualityDisplayWidget *>(_o);
        Q_UNUSED(_t)
        switch (_id) {
        case 0: _t->exitThreadAvailable(); break;
        case 1: _t->labelInitalizeAvailable(); break;
        case 2: _t->qualityScoresAvailable((*reinterpret_cast< QVariant(*)>(_a[1]))); break;
        case 3: _t->setCurrentViewName((*reinterpret_cast< const QString(*)>(_a[1]))); break;
        case 4: _t->setCurrentViewNameImage((*reinterpret_cast< const QString(*)>(_a[1])),(*reinterpret_cast< QImage(*)>(_a[2]))); break;
        case 5: _t->setCurrentViewNameVideo((*reinterpret_cast< const QString(*)>(_a[1])),(*reinterpret_cast< QVariant(*)>(_a[2]))); break;
        case 6: _t->setLabelInitialize((*reinterpret_cast< const QString(*)>(_a[1]))); break;
        case 7: _t->setViewUncomplete((*reinterpret_cast< const QString(*)>(_a[1]))); break;
        case 8: _t->setQualityScores((*reinterpret_cast< QString(*)>(_a[1])),(*reinterpret_cast< QVariant(*)>(_a[2]))); break;
        default: ;
        }
    } else if (_c == QMetaObject::IndexOfMethod) {
        int *result = reinterpret_cast<int *>(_a[0]);
        {
            using _t = void (QualityDisplayWidget::*)();
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&QualityDisplayWidget::exitThreadAvailable)) {
                *result = 0;
                return;
            }
        }
        {
            using _t = void (QualityDisplayWidget::*)();
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&QualityDisplayWidget::labelInitalizeAvailable)) {
                *result = 1;
                return;
            }
        }
        {
            using _t = void (QualityDisplayWidget::*)(QVariant );
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&QualityDisplayWidget::qualityScoresAvailable)) {
                *result = 2;
                return;
            }
        }
    }
}

QT_INIT_METAOBJECT const QMetaObject QualityDisplayWidget::staticMetaObject = { {
    QMetaObject::SuperData::link<QWidget::staticMetaObject>(),
    qt_meta_stringdata_QualityDisplayWidget.data,
    qt_meta_data_QualityDisplayWidget,
    qt_static_metacall,
    nullptr,
    nullptr
} };


const QMetaObject *QualityDisplayWidget::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *QualityDisplayWidget::qt_metacast(const char *_clname)
{
    if (!_clname) return nullptr;
    if (!strcmp(_clname, qt_meta_stringdata_QualityDisplayWidget.stringdata0))
        return static_cast<void*>(this);
    return QWidget::qt_metacast(_clname);
}

int QualityDisplayWidget::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QWidget::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 9)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 9;
    } else if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        if (_id < 9)
            *reinterpret_cast<int*>(_a[0]) = -1;
        _id -= 9;
    }
    return _id;
}

// SIGNAL 0
void QualityDisplayWidget::exitThreadAvailable()
{
    QMetaObject::activate(this, &staticMetaObject, 0, nullptr);
}

// SIGNAL 1
void QualityDisplayWidget::labelInitalizeAvailable()
{
    QMetaObject::activate(this, &staticMetaObject, 1, nullptr);
}

// SIGNAL 2
void QualityDisplayWidget::qualityScoresAvailable(QVariant _t1)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(std::addressof(_t1))) };
    QMetaObject::activate(this, &staticMetaObject, 2, _a);
}
QT_WARNING_POP
QT_END_MOC_NAMESPACE
