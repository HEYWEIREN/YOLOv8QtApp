// 包含主窗口类的头文件
#include "YOLOv8QtApp.h"
// 包含 Qt Designer 自动生成的 UI 类
#include "ui_YOLOv8QtApp.h"
// Qt 文件对话框与提示框
#include <QFileDialog>
#include <QMessageBox>
#include <QDir>

// 构造函数：初始化界面和模型封装器
YOLOv8QtApp::YOLOv8QtApp(QWidget *parent)
    : QMainWindow(parent), ui(new Ui::YOLOv8QtApp) {

    ui->setupUi(this);  // 将 ui 布局和控件与类对象进行绑定

    // 自动创建输出结果保存目录（outputs）
    QDir().mkdir("outputs");

    // 创建模型封装器（推理入口）
    wrapper_ = std::make_unique<YoloWrapper>();
}

// 析构函数：释放 UI 资源
YOLOv8QtApp::~YOLOv8QtApp() {
    delete ui;
}

// [按钮响应] 打开图片
void YOLOv8QtApp::on_btnOpenImage_clicked() {
    // 弹出文件选择框
    QString fileName = QFileDialog::getOpenFileName(this, "选择图片", "", "Images (*.png *.jpg *.jpeg)");
    if (fileName.isEmpty()) return;  // 用户取消

    // 读取图像为 OpenCV 格式
    currentImage_ = cv::imread(fileName.toStdString());

    // 失败处理
    if (currentImage_.empty()) {
        QMessageBox::warning(this, "错误", "图像加载失败！");
        return;
    }

    // 显示原图并保存
    displayImage(currentImage_, "origin.jpg");
}

// [按钮响应] 执行目标检测
void YOLOv8QtApp::on_btnDetect_clicked() {
    // 检查封装器是否存在
    if (!wrapper_) {
        QMessageBox::warning(this, "错误", "模型封装未初始化！");
        return;
    }

    // 检查是否加载图像
    if (currentImage_.empty()) {
        QMessageBox::warning(this, "错误", "请先打开图片！");
        return;
    }

    // 加载检测模型（你需要将路径替换为自己的模型路径）
    if (!wrapper_->loadModel("C:/Users/31268/Documents/YOLOv8QtApp/models/yolov8l.onnx", ModelType::DETECT)) {
        QMessageBox::warning(this, "错误", "检测模型加载失败！");
        return;
    }

    // 执行推理
    cv::Mat result = wrapper_->infer(currentImage_);
    if (result.empty()) {
        QMessageBox::warning(this, "错误", "推理失败！");
        return;
    }

    // 显示并保存结果
    displayImage(result, "detect.jpg");
}

// [按钮响应] 执行实例分割
void YOLOv8QtApp::on_btnSegment_clicked() {
    if (!wrapper_) {
        QMessageBox::warning(this, "错误", "模型封装未初始化！");
        return;
    }

    if (currentImage_.empty()) {
        QMessageBox::warning(this, "错误", "请先打开图片！");
        return;
    }

    if (!wrapper_->loadModel("C:/Users/31268/Documents/YOLOv8QtApp/models/yolov8l-seg.onnx", ModelType::SEGMENT)) {
        QMessageBox::warning(this, "错误", "分割模型加载失败！");
        return;
    }

    cv::Mat result = wrapper_->infer(currentImage_);
    if (result.empty()) {
        QMessageBox::warning(this, "错误", "推理失败！");
        return;
    }

    displayImage(result, "segment.jpg");
}

// [按钮响应] 执行姿态估计
void YOLOv8QtApp::on_btnPose_clicked() {
    if (!wrapper_) {
        QMessageBox::warning(this, "错误", "模型封装未初始化！");
        return;
    }

    if (currentImage_.empty()) {
        QMessageBox::warning(this, "错误", "请先打开图片！");
        return;
    }

    if (!wrapper_->loadModel("C:/Users/31268/Documents/YOLOv8QtApp/models/yolov8l-pose.onnx", ModelType::POSE)) {
        QMessageBox::warning(this, "错误", "姿态模型加载失败！");
        return;
    }

    cv::Mat result = wrapper_->infer(currentImage_);
    if (result.empty()) {
        QMessageBox::warning(this, "错误", "推理失败！");
        return;
    }

    displayImage(result, "pose.jpg");
}


// [按钮响应] 执行分类（先接通按钮，业务后续接入）
void YOLOv8QtApp::on_btnClassify_clicked()
{
    if (!wrapper_) { QMessageBox::warning(this, "错误", "模型未初始化"); return; }
    if (currentImage_.empty()) { QMessageBox::warning(this, "错误", "请先打开图片"); return; }

    // 1) 如果没有检测结果，就先跑一次 detect（保证有数据可汇总）
    if (wrapper_->currentType() != ModelType::DETECT || wrapper_->lastResults().empty()) {
        const std::string detectPath = "C:/Users/31268/Documents/YOLOv8QtApp/models/yolov8l.onnx"; // <- 改成你的
        if (!wrapper_->loadModel(detectPath, ModelType::DETECT)) {
            QMessageBox::warning(this, "错误", "检测模型加载失败");
            return;
        }
        cv::Mat vis = wrapper_->infer(currentImage_);
        if (vis.empty()) { QMessageBox::warning(this, "错误", "推理失败"); return; }
        displayImage(vis, "classify_detect_vis.jpg"); // 可留可删
    }

    // 2) 汇总类别
    const auto& results = wrapper_->lastResults();
    if (results.empty()) {
        QMessageBox::information(this, "分类结果", "未检测到任何目标。");
        return;
    }

    const float thr = 0.25f;
    std::map<int, int> cnt;
    std::map<int, float> best;
    for (const auto& r : results) {
        if (r.conf < thr) continue;
        cnt[r.class_idx] += 1;
        if (!best.count(r.class_idx) || r.conf > best[r.class_idx]) best[r.class_idx] = r.conf;
    }

    if (cnt.empty()) {
        QMessageBox::information(this, "分类结果", QString("检测结果均低于阈值 %.2f").arg(thr));
        return;
    }

    // 3) 类别名映射（你现在 getNames() 是 unordered_map）
    const auto& names = wrapper_->getNames();
    QStringList lines;
    for (const auto& kv : cnt) {
        int cls = kv.first;
        int c = kv.second;

        QString name = QString::number(cls);
        auto it = names.find(cls);
        if (it != names.end()) name = QString::fromStdString(it->second);

        lines << QString("%1  x%2  (max=%3)")
                     .arg(name)
                     .arg(c)
                     .arg(best[cls], 0, 'f', 2);

    }

    // 4) 展示
    QMessageBox::information(this, "这张图包含的类别", lines.join("\n"));

    cv::Mat out = currentImage_.clone();
    int y = 40;
    for (const auto& s : lines) {
        cv::putText(out, s.toStdString(), cv::Point(20, y),
                    cv::FONT_HERSHEY_SIMPLEX, 0.9, cv::Scalar(255,255,255), 2);
        y += 32;
    }
    displayImage(out, "classify_summary.jpg");
}


// [按钮响应] 执行旋转框 OBB（先接通按钮，业务后续接入）
void YOLOv8QtApp::on_btnOBB_clicked()
{
    if (!wrapper_) { QMessageBox::warning(this, "错误", "模型封装未初始化！"); return; }
    if (currentImage_.empty()) { QMessageBox::warning(this, "错误", "请先打开图片！"); return; }

    const std::string obbPath = "C:/Users/31268/Documents/YOLOv8QtApp/models/yolov8n-obb.onnx";
    if (!wrapper_->loadModel(obbPath, ModelType::OBB)) {
        QMessageBox::warning(this, "错误", "OBB 模型加载失败！");
        return;
    }

    cv::Mat out = wrapper_->infer(currentImage_);
    if (out.empty()) { QMessageBox::warning(this, "错误", "推理失败！"); return; }

    displayImage(out, "obb.jpg");
}


// [按钮响应] 执行跟踪（你说先放一边）
void YOLOv8QtApp::on_btnTrack_clicked()
{
    QMessageBox::information(this, "Track", "Track 暂未实现（后续接入视频流）");
}



// 显示图像到 QLabel，并保存图像文件
void YOLOv8QtApp::displayImage(const cv::Mat& mat, const QString& saveName) {
    if (mat.empty()) return;

    // 保存推理结果图像到 outputs 文件夹
    cv::imwrite("outputs/" + saveName.toStdString(), mat);

    // 转换为 Qt 图像格式并显示在 QLabel 中
    QImage img = MatToQImage(mat);
    ui->labelImage->setPixmap(QPixmap::fromImage(img).scaled(
        ui->labelImage->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
}

// 将 OpenCV 图像格式（Mat）转换为 Qt 显示格式（QImage）
QImage YOLOv8QtApp::MatToQImage(const cv::Mat& mat) {
    if (mat.type() == CV_8UC3) {
        QImage image((uchar*)mat.data, mat.cols, mat.rows, mat.step, QImage::Format_RGB888);
        return image.rgbSwapped();  // BGR → RGB
    } else if (mat.type() == CV_8UC1) {
        return QImage(mat.data, mat.cols, mat.rows, mat.step, QImage::Format_Grayscale8);
    }
    return QImage();  // 不支持的格式返回空图
}
