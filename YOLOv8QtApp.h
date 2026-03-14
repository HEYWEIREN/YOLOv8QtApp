#pragma once  // 只包含一次此头文件，防止重复定义

// Qt 相关头文件
#include <QMainWindow>  // 提供主窗口类
#include <QString>      // 字符串类
#include <QImage>       // 图像类

// OpenCV 图像处理库
#include <opencv2/opencv.hpp>

// 模型封装类头文件（我们自己写的推理包装器）
#include "yolo_wrapper.h"  // 加入封装头文件，定义了 YoloWrapper 类

// Qt UI 命名空间声明（自动生成的 ui 文件会使用此命名空间）
QT_BEGIN_NAMESPACE
namespace Ui { class YOLOv8QtApp; }
QT_END_NAMESPACE

// 主窗口类，继承自 QMainWindow，提供 UI 控件响应能力
class YOLOv8QtApp : public QMainWindow {
    Q_OBJECT  // 启用 Qt 元对象系统（信号/槽机制的宏）

public:
    // 构造函数，创建主窗口
    YOLOv8QtApp(QWidget *parent = nullptr);
    // 析构函数，释放资源
    ~YOLOv8QtApp();

private slots:
    // 以下槽函数响应用户在 UI 上点击对应按钮的行为
    void on_btnOpenImage_clicked();  // 打开图像按钮
    void on_btnDetect_clicked();     // 检测按钮
    void on_btnSegment_clicked();    // 分割按钮
    void on_btnPose_clicked();       // 姿态估计按钮
    void on_btnClassify_clicked();
    void on_btnOBB_clicked();
    void on_btnTrack_clicked();

private:
    Ui::YOLOv8QtApp *ui;             // 指向界面控件的指针，指向 Qt Designer 生成的类
    cv::Mat currentImage_;          // 当前加载的图像，供模型推理使用

    // 将 OpenCV 的 Mat 转换为 Qt 的 QImage，方便在 QLabel 上显示
    QImage MatToQImage(const cv::Mat& mat);
    // 显示图像到 QLabel，并保存图像到 outputs 目录
    void displayImage(const cv::Mat& mat, const QString& saveName);

    // 模型封装器指针（智能指针） → 用于加载不同类型的模型并推理
    std::unique_ptr<YoloWrapper> wrapper_;

};

