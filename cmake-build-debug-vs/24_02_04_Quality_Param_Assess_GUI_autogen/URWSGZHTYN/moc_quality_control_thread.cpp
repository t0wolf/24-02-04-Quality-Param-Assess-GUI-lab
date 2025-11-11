/****************************************************************************
** Meta object code from reading C++ file 'quality_control_thread.h'
**
** Created by: The Qt Meta Object Compiler version 67 (Qt 5.15.2)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include <memory>
#include "../../../process_threads/quality_control_thread.h"
#include <QtCore/qbytearray.h>
#include <QtCore/qmetatype.h>
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'quality_control_thread.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 67
#error "This file was generated using the moc from 5.15.2. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
QT_WARNING_PUSH
QT_WARNING_DISABLE_DEPRECATED
struct qt_meta_stringdata_QualityControlThread_t {
    QByteArrayData data[11];
    char stringdata0[123];
};
#define QT_MOC_LITERAL(idx, ofs, len) \
    Q_STATIC_BYTE_ARRAY_DATA_HEADER_INITIALIZER_WITH_OFFSET(len, \
    qptrdiff(offsetof(qt_meta_stringdata_QualityControlThread_t, stringdata0) + ofs \
        - idx * sizeof(QByteArrayData)) \
    )
static const qt_meta_stringdata_QualityControlThread_t qt_meta_stringdata_QualityControlThread = {
    {
QT_MOC_LITERAL(0, 0, 20), // "QualityControlThread"
QT_MOC_LITERAL(1, 21, 10), // "sigRoIRect"
QT_MOC_LITERAL(2, 32, 0), // ""
QT_MOC_LITERAL(3, 33, 5), // "qRect"
QT_MOC_LITERAL(4, 39, 14), // "sigVideoResult"
QT_MOC_LITERAL(5, 54, 7), // "qResult"
QT_MOC_LITERAL(6, 62, 15), // "setQualityInput"
QT_MOC_LITERAL(7, 78, 9), // "qViewName"
QT_MOC_LITERAL(8, 88, 11), // "qVideoClips"
QT_MOC_LITERAL(9, 100, 14), // "qKeyframeIdxes"
QT_MOC_LITERAL(10, 115, 7) // "fRadius"

    },
    "QualityControlThread\0sigRoIRect\0\0qRect\0"
    "sigVideoResult\0qResult\0setQualityInput\0"
    "qViewName\0qVideoClips\0qKeyframeIdxes\0"
    "fRadius"
};
#undef QT_MOC_LITERAL

static const uint qt_meta_data_QualityControlThread[] = {

 // content:
       8,       // revision
       0,       // classname
       0,    0, // classinfo
       3,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       2,       // signalCount

 // signals: name, argc, parameters, tag, flags
       1,    1,   29,    2, 0x06 /* Public */,
       4,    1,   32,    2, 0x06 /* Public */,

 // slots: name, argc, parameters, tag, flags
       6,    4,   35,    2, 0x0a /* Public */,

 // signals: parameters
    QMetaType::Void, QMetaType::QVariant,    3,
    QMetaType::Void, QMetaType::QVariant,    5,

 // slots: parameters
    QMetaType::Void, QMetaType::QString, QMetaType::QVariant, QMetaType::QVariant, QMetaType::Float,    7,    8,    9,   10,

       0        // eod
};

void QualityControlThread::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        auto *_t = static_cast<QualityControlThread *>(_o);
        Q_UNUSED(_t)
        switch (_id) {
        case 0: _t->sigRoIRect((*reinterpret_cast< QVariant(*)>(_a[1]))); break;
        case 1: _t->sigVideoResult((*reinterpret_cast< QVariant(*)>(_a[1]))); break;
        case 2: _t->setQualityInput((*reinterpret_cast< QString(*)>(_a[1])),(*reinterpret_cast< QVariant(*)>(_a[2])),(*reinterpret_cast< QVariant(*)>(_a[3])),(*reinterpret_cast< float(*)>(_a[4]))); break;
        default: ;
        }
    } else if (_c == QMetaObject::IndexOfMethod) {
        int *result = reinterpret_cast<int *>(_a[0]);
        {
            using _t = void (QualityControlThread::*)(QVariant );
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&QualityControlThread::sigRoIRect)) {
                *result = 0;
                return;
            }
        }
        {
            using _t = void (QualityControlThread::*)(QVariant );
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&QualityControlThread::sigVideoResult)) {
                *result = 1;
                return;
            }
        }
    }
}

QT_INIT_METAOBJECT const QMetaObject QualityControlThread::staticMetaObject = { {
    QMetaObject::SuperData::link<QThread::staticMetaObject>(),
    qt_meta_stringdata_QualityControlThread.data,
    qt_meta_data_QualityControlThread,
    qt_static_metacall,
    nullptr,
    nullptr
} };


const QMetaObject *QualityControlThread::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *QualityControlThread::qt_metacast(const char *_clname)
{
    if (!_clname) return nullptr;
    if (!strcmp(_clname, qt_meta_stringdata_QualityControlThread.stringdata0))
        return static_cast<void*>(this);
    return QThread::qt_metacast(_clname);
}

int QualityControlThread::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QThread::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 3)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 3;
    } else if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        if (_id < 3)
            *reinterpret_cast<int*>(_a[0]) = -1;
        _id -= 3;
    }
    return _id;
}

// SIGNAL 0
void QualityControlThread::sigRoIRect(QVariant _t1)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(std::addressof(_t1))) };
    QMetaObject::activate(this, &staticMetaObject, 0, _a);
}

// SIGNAL 1
void QualityControlThread::sigVideoResult(QVariant _t1)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(std::addressof(_t1))) };
    QMetaObject::activate(this, &staticMetaObject, 1, _a);
}
QT_WARNING_POP
QT_END_MOC_NAMESPACE
