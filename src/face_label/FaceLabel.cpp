#include "FaceLabel.h"
#include <iostream>
#include <iomanip>
#include <filesystem>
#include <sys/stat.h>

namespace fs = std::filesystem;

namespace TA::FaceLabel {

    FaceLabel::FaceLabel(const std::string& yunet_path, const std::string& emotion_onnx_path)
        : face_detector_(yunet_path) 
    {
        // Load model ONNX ViT
        emotion_net_ = cv::dnn::readNetFromONNX(emotion_onnx_path);
        emotion_net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        emotion_net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
        
        // Opsional: Jika Anda punya GPU NVIDIA dan OpenCV dikompilasi dengan CUDA, gunakan ini:
        // emotion_net_.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
    }

    cv::Mat FaceLabel::PreprocessEmotion(const cv::Mat& face_img) {
        // 1. Ukuran ViT adalah 224x224
        cv::Mat resized;
        cv::resize(face_img, resized, cv::Size(224, 224), 0, 0, cv::INTER_LINEAR);
        
        // 2. Normalisasi (Mean 0.5, Std 0.5)
        // Karena blobFromImage tidak bisa handle mean/std berbeda per channel secara langsung 
        // dengan mudah, kita bisa "akali" dengan scalar 0.5.
        // Rumus: (pixel/255 - 0.5) / 0.5  ->  (pixel/255 - 0.5) * 2
        
        // cv::dnn::blobFromImage sudah melakukan: (image - mean) * scalefactor
        // Jadi kita set scalefactor = 1/127.5 (karena 255 * 0.5 = 127.5)
        // dan mean = [127.5, 127.5, 127.5]
        cv::Mat blob = cv::dnn::blobFromImage(
            resized, 
            1.0 / 127.5, 
            cv::Size(224, 224), 
            cv::Scalar(127.5, 127.5, 127.5), 
            true,   // swapRB = true (OpenCV defaultnya BGR, ViT butuh RGB)
            false   // crop = false
        );

        return blob;
    }

    void FaceLabel::SaveDataset(const cv::Mat& face_img, const std::string& label, 
                                const std::string& video_name, double timestamp, float score,
                                std::ofstream& csv_file, const std::string& out_dir) 
    {
        std::string class_dir = out_dir + "/" + label;
        if (!fs::exists(class_dir)) fs::create_directories(class_dir);

        std::ostringstream filename;
        filename << video_name << "_" << std::fixed << std::setprecision(2) << timestamp 
                 << "_" << std::setprecision(2) << score << ".jpg";
        
        std::string full_path = class_dir + "/" + filename.str();
        
        cv::imwrite(full_path, face_img);

        csv_file << full_path << "," << video_name << "," 
                 << timestamp << "," << score << "\n";
    }

    void FaceLabel::ProcessVideos(const std::vector<VideoConfig>& videos, const std::string& output_dataset_dir) {
        std::ofstream csv_file(output_dataset_dir + "/metadata.csv");
        csv_file << "filepath,video_source,timestamp_sec,detection_score\n";

        for (const auto& vid_cfg : videos) {
            cv::VideoCapture cap(vid_cfg.path, cv::CAP_FFMPEG);
            if (!cap.isOpened()) {
                std::cerr << "Gagal buka video: " << vid_cfg.path << "\n";
                continue;
            }

            double fps = cap.get(cv::CAP_PROP_FPS);
            if (fps <= 0) fps = 30.0;

            int frame_jump = static_cast<int>(vid_cfg.interval_sec * fps);
            if (frame_jump < 1) frame_jump = 1;

            std::string vid_name = fs::path(vid_cfg.path).stem().string();
            std::cout << "Memproses: " << vid_name << " (FPS: " << fps << ", Jump: " << frame_jump << " frames)\n";

            cv::Mat frame;
            int frame_count = 0;

            while (cap.read(frame)) {
                if (frame_count % frame_jump == 0) {
                    // Deteksi Wajah (Minimal 50px agar hasil crop tidak terlalu blur saat di-resize ke 224x224)
                    auto faces = face_detector_.Run(frame, 50); 

                    double current_time = frame_count / fps;

                    for (const auto& face : faces) {
                        cv::Mat cropped = FaceExtractor::CropFace(frame, face);
                        if (cropped.empty()) continue;

                        cv::Mat blob = PreprocessEmotion(cropped);
                        emotion_net_.setInput(blob);
                        cv::Mat output = emotion_net_.forward(); // Output 1x7

                        // ViT biasanya sudah mengeluarkan probabilitas (softmax di dalam model),
                        // tapi minMaxLoc tetap aman untuk mencari nilai tertinggi.
                        cv::Point max_loc;
                        cv::minMaxLoc(output, nullptr, nullptr, nullptr, &max_loc);

                        int class_id = max_loc.x;
                        
                        // Pastikan ID tidak out of bounds
                        if (class_id >= 0 && class_id < static_cast<int>(emotions_.size())) {
                            std::string label = emotions_[class_id];
                            SaveDataset(cropped, label, vid_name, current_time, face.score, csv_file, output_dataset_dir);
                        }
                    }
                }
                frame_count++;
            }
            cap.release();
            std::cout << "Selesai: " << vid_name << "\n";
        }
        csv_file.close();
    }
}