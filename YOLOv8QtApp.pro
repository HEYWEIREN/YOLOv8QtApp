QT += core gui widgets

CONFIG += c++17

INCLUDEPATH += C:/OpenCV/opencv/build/include
LIBS += -LC:/OpenCV/opencv/build/x64/vc16/lib \
    -lopencv_world480

INCLUDEPATH += C:/Users/31268/Desktop/yolov8-onnx-cpp-main/onnxruntime-win-x64-1.15.1/include
LIBS += -LC:/Users/31268/Desktop/yolov8-onnx-cpp-main/onnxruntime-win-x64-1.15.1/lib \
    -lonnxruntime

INCLUDEPATH += "C:/Users/31268/Desktop/yolov8-onnx-cpp-main/onnxruntime-win-x64-1.15.1/include"

FORMS += YOLOv8QtApp.ui
#RESOURCES += YOLOv8QtApp.qrc

SOURCES += \
    main.cpp \
    YOLOv8QtApp.cpp \
    autobackend.cpp \
    augment.cpp \
    common.cpp \
    onnx_model_base.cpp \
    ops.cpp \
    yolo_wrapper.cpp

HEADERS += \
    YOLOv8QtApp.h \
    autobackend.h \
    augment.h \
    common.h \
    constants.h \
    onnx_model_base.h \
    ops.h \
    yolo_wrapper.h

RESOURCES += \
    resources.qrc
