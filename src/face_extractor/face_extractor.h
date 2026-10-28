#pragma once
#include <opencv2/opencv.hpp>
#include "inference/inference.h"

namespace TA::FaceLabel {
    class FaceExtractor {
    public:
        // Fungsi statis: input frame asli & box, return gambar yang udah di-crop
        static cv::Mat CropFace(const cv::Mat& frame, const Inference::FaceResult& face);
    };
}