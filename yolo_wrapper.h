#ifndef YOLO_WRAPPER_H
#define YOLO_WRAPPER_H

// OpenCV 图像处理库
#include <opencv2/opencv.hpp>

// 智能指针、字符串、向量容器
#include <memory>
#include <string>
#include <vector>
#include <unordered_map>


// Qt 的调试输出模块
#include <QDebug>

// ONNX 后端和模型基类
#include "autobackend.h"
#include "onnx_model_base.h"

// 模型任务类型枚举（UI 用）
enum class ModelType {
    DETECT,
    SEGMENT,
    POSE,
    CLASSIFY,
    OBB,
    TRACK
};

// 封装模型的推理类，对 UI 层隐藏所有底层细节
class YoloWrapper {
private:
    std::vector<YoloResults> last_results_;

public:
    const std::vector<YoloResults>& lastResults() const { return last_results_; }

    const std::unordered_map<int, std::string>& getNames() const {
        static const std::unordered_map<int, std::string> empty;
        return model_ ? model_->getNames() : empty;
    }
    ModelType currentType() const { return type_; }


public:
    /**
     * @brief 加载模型
     * @param path ONNX 模型路径
     * @param type 模型类型（检测 / 分割 / 姿态）
     * @param inputW 输入宽度（默认 640）
     * @param inputH 输入高度（默认 640）
     * @return 加载是否成功
     */
    bool loadModel(const std::string& path, ModelType type, int inputW = 640, int inputH = 640) {
        type_ = type;  // 保存当前模型类型
        model_ = std::make_unique<AutoBackendOnnx>(path.c_str(), "qt_log", "cpu");  // 创建 ONNX 后端
        last_results_.clear();
        return model_ != nullptr;
    }

    /**
     * @brief 对输入图像执行推理并绘制结果
     * @param inputImg 输入图像（OpenCV 格式）
     * @return 绘制好检测框 / 掩码 / 姿态的图像
     */
    cv::Mat infer(const cv::Mat& inputImg) {
        // 检查模型和图像是否正常
        if (!model_ || inputImg.empty()) return inputImg;


        float conf = 0.3f, iou = 0.45f, mask_thresh = 0.5f;
        if (type_ == ModelType::OBB) {
            conf = 0.55f;
            iou = 0.25f;
        }

        // 执行模型推理，返回检测结果（每个目标的结构体）
        cv::Mat img = inputImg.clone();  //  不用 const_cast，避免后端改原图
        std::vector<YoloResults> results = model_->predict_once(img, conf, iou, mask_thresh);
        last_results_ = results;         //  正确：函数调用结束后再缓存

        // 拷贝图像用于绘制
        cv::Mat output = inputImg.clone();

        // 常见类别名称（用于标签）
        std::vector<std::string> classNames = {
            "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck",
            "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
            "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
            "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
            "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
            "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
            "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
            "donut", "cake", "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet",
            "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
            "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
            "teddy bear", "hair drier", "toothbrush"
        };

        // 为每个目标生成颜色（实例唯一颜色）
        std::vector<cv::Scalar> colors;
        for (int i = 0; i < 80; ++i)
            colors.emplace_back((i * 37) % 255, (i * 17) % 255, (i * 29) % 255);

        // 遍历所有检测结果进行绘图
        for (size_t i = 0; i < results.size(); ++i) {
            const auto& res = results[i];  // 当前目标

            // 输出调试信息（包括框、掩膜大小等）
            qDebug().noquote() << QString("bbox %1: [x=%2, y=%3, w=%4, h=%5], conf=%6, hasMask=%7, nonZero=%8, maskSize=[%9x%10], equalToBbox=%11")
                                      .arg(i)
                                      .arg((int)res.bbox.x)
                                      .arg((int)res.bbox.y)
                                      .arg((int)res.bbox.width)
                                      .arg((int)res.bbox.height)
                                      .arg(res.conf, 0, 'f', 3)
                                      .arg(!res.mask.empty())
                                      .arg(res.mask.empty() ? -1 : cv::countNonZero(res.mask))
                                      .arg(res.mask.cols)
                                      .arg(res.mask.rows)
                                      .arg((res.mask.cols == (int)res.bbox.width && res.mask.rows == (int)res.bbox.height) ? "true" : "false");

            int cls = res.class_idx;     // 类别索引
            float score = res.conf;      // 置信度

            // 使用 HSV 转 BGR 生成明亮高饱和颜色
            cv::Mat hsv(1, 1, CV_8UC3, cv::Scalar((i * 47) % 180, 200, 255));
            cv::Mat bgr;
            cv::cvtColor(hsv, bgr, cv::COLOR_HSV2BGR);
            cv::Scalar color = bgr.at<cv::Vec3b>(0, 0);

            // 绘制检测框与类别标签
            std::string label = classNames[cls] + " " + std::to_string(score).substr(0, 4);
            int base = 0;
            auto ts = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &base);
            cv::Point org(res.bbox.x, res.bbox.y - 5);
            if (org.y < ts.height) org.y = res.bbox.y + ts.height + 5;
            if (!res.has_rbox) {
                cv::rectangle(output, { org.x, org.y - ts.height, ts.width, ts.height + base }, color, cv::FILLED);
            }
            cv::putText(output, label, org, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);

            // 绘制分割掩码
            if (type_ == ModelType::SEGMENT && !res.mask.empty()) {
                cv::Rect box = (cv::Rect)res.bbox & cv::Rect(0, 0, output.cols, output.rows);
                if (res.mask.size() == cv::Size(box.width, box.height)) {
                    cv::Mat roi = output(box);
                    cv::Mat colorMask(roi.size(), CV_8UC3, color);
                    cv::Mat blended;
                    cv::addWeighted(roi, 1.0, colorMask, 0.4, 0, blended);
                    cv::Mat binMask;
                    res.mask.convertTo(binMask, CV_8U);  // 掩码转换为8位
                    blended.copyTo(roi, binMask);
                } else {
                    qDebug() << "[warn] mask size != bbox size:" << res.mask.size().width << "x" << res.mask.size().height
                             << "vs" << box.width << "x" << box.height;
                }
            }

            // 绘制姿态关键点和骨架
            if (type_ == ModelType::POSE && !res.keypoints.empty()) {
                drawPose(output, res);
            }

            if (res.has_rbox) {
                cv::Point2f pts[4];
                res.rbox.points(pts);
                for (int k = 0; k < 4; ++k)
                    cv::line(output, pts[k], pts[(k + 1) % 4], color, 2);
            } else {
                cv::rectangle(output, res.bbox, color, 2);
            }
        }

        return output;
    }

private:
    // 智能指针保存 ONNX 模型后端
    std::unique_ptr<AutoBackendOnnx> model_;
    // 当前模型类型（检测 / 分割 / 姿态）
    ModelType type_;

    // 姿态关键点绘图函数
    void drawPose(cv::Mat& img, const YoloResults& res) {
        // 定义骨架连接关系（关键点对）
        static std::vector<std::vector<int>> skeleton = {
            {16,14},{14,12},{17,15},{15,13},{12,13},{6,12},{7,13},{6,7},
            {6,8},{7,9},{8,10},{9,11},{2,3},{1,2},{1,3},{2,4},{3,5},{4,6},{5,7}
        };

        // 绘制关键点（17个点）
        for (int i = 0; i < 17; ++i) {
            int idx = i * 3;
            float x = res.keypoints[idx];
            float y = res.keypoints[idx + 1];
            float c = res.keypoints[idx + 2];
            if (c > 0.5f)
                cv::circle(img, { (int)x, (int)y }, 3, cv::Scalar(0, 0, 255), -1);
        }

        // 绘制骨架连线（连接两个关键点）
        for (const auto& sk : skeleton) {
            int i1 = sk[0] - 1, i2 = sk[1] - 1;
            float x1 = res.keypoints[i1 * 3];
            float y1 = res.keypoints[i1 * 3 + 1];
            float c1 = res.keypoints[i1 * 3 + 2];
            float x2 = res.keypoints[i2 * 3];
            float y2 = res.keypoints[i2 * 3 + 1];
            float c2 = res.keypoints[i2 * 3 + 2];
            if (c1 > 0.5f && c2 > 0.5f)
                cv::line(img, { (int)x1, (int)y1 }, { (int)x2, (int)y2 }, cv::Scalar(0, 255, 255), 2);
        }
    }

};



#endif // YOLO_WRAPPER_H
