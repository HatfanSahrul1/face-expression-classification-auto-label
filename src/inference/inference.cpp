#include "inference.h"
#include <stdexcept>

namespace TA::FaceLabel {
    Inference::Inference(const std::string& model_path) { InitModel(model_path); }

    void Inference::InitModel(const std::string& model_path) {
        detector_ = cv::FaceDetectorYN::create(model_path, "", cv::Size(320, 320), 0.9f, 0.3f, 5000);
        if (!detector_) throw std::runtime_error("Gagal load YuNet model");
    }

    std::vector<Inference::FaceResult> Inference::Run(const cv::Mat& frame, int min_face_size) {
        std::vector<FaceResult> results;
        if (frame.empty() || !detector_) return results;

        detector_->setInputSize(frame.size());
        cv::Mat faces;
        detector_->detect(frame, faces);

        for (int i = 0; i < faces.rows; ++i) {
            FaceResult r;
            r.box.x = static_cast<int>(faces.at<float>(i, 0));
            r.box.y = static_cast<int>(faces.at<float>(i, 1));
            r.box.width = static_cast<int>(faces.at<float>(i, 2));
            r.box.height = static_cast<int>(faces.at<float>(i, 3));
            
            for (int k = 0; k < 5; ++k) {
                r.landmarks[k].x = faces.at<float>(i, 4 + k * 2);
                r.landmarks[k].y = faces.at<float>(i, 5 + k * 2);
            }
            r.score = faces.at<float>(i, 14);
            r.box &= cv::Rect(0, 0, frame.cols, frame.rows);

            if (min_face_size > 0 && (r.box.width < min_face_size || r.box.height < min_face_size)) continue;
            results.push_back(r);
        }
        return results;
    }
}