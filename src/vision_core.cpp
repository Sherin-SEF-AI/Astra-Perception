#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <numeric>

namespace py = pybind11;

class VisionAccelerator {
public:
    VisionAccelerator() {}

    // Drivable Area Detection
    py::object process_drivable_area(py::array_t<uint8_t> input_image, float scale, float dist_thresh, float tex_thresh) {
        // Convert numpy array to cv::Mat
        auto ref = input_image.unchecked<3>();
        cv::Mat image(ref.shape(0), ref.shape(1), CV_8UC3, (void*)ref.data(0, 0, 0));
        
        int h = image.rows;
        int w = image.cols;
        
        cv::Mat small;
        cv::resize(image, small, cv::Size(), scale, scale, cv::INTER_NEAREST);
        
        int sh = small.rows;
        int sw = small.cols;
        
        // Sampling
        std::vector<cv::Rect> rois = {
            cv::Rect(sw*0.45, sh*0.75, sw*0.1, sh*0.1),
            cv::Rect(sw*0.30, sh*0.70, sw*0.1, sh*0.1),
            cv::Rect(sw*0.60, sh*0.70, sw*0.1, sh*0.1)
        };
        
        std::vector<double> l_vals, a_vals, b_vals;
        cv::Mat small_lab;
        cv::cvtColor(small, small_lab, cv::COLOR_BGR2Lab);
        
        for (const auto& roi : rois) {
            if (roi.x >= 0 && roi.y >= 0 && roi.x + roi.width <= sw && roi.y + roi.height <= sh) {
                cv::Mat sample = small_lab(roi);
                cv::Scalar mean = cv::mean(sample);
                l_vals.push_back(mean[0]);
                a_vals.push_back(mean[1]);
                b_vals.push_back(mean[2]);
            }
        }
        
        if (l_vals.empty()) return py::none();
        
        double ml = std::accumulate(l_vals.begin(), l_vals.end(), 0.0) / l_vals.size();
        double ma = std::accumulate(a_vals.begin(), a_vals.end(), 0.0) / a_vals.size();
        double mb = std::accumulate(b_vals.begin(), b_vals.end(), 0.0) / b_vals.size();
        
        // Texture calculation
        cv::Mat gray;
        cv::extractChannel(small_lab, gray, 0);
        cv::Mat sobelx, sobely;
        cv::Sobel(gray, sobelx, CV_32F, 1, 0, 3);
        cv::Sobel(gray, sobely, CV_32F, 0, 1, 3);
        cv::Mat texture = cv::abs(sobelx) + cv::abs(sobely);
        
        // Final mask
        cv::Mat mask = cv::Mat::zeros(sh, sw, CV_8U);
        for (int r = 0; r < sh; ++r) {
            for (int c = 0; c < sw; ++c) {
                cv::Vec3b pix = small_lab.at<cv::Vec3b>(r, c);
                float d = 0.2f * std::abs(pix[0] - (float)ml) + 1.2f * std::abs(pix[1] - (float)ma) + 1.2f * std::abs(pix[2] - (float)mb);
                float t = texture.at<float>(r, c);
                
                if (d < dist_thresh && t < tex_thresh && r > sh*0.45) {
                    mask.at<uchar>(r, c) = 255;
                }
            }
        }
        
        // Morphological and Connectivity
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
        cv::morphologyEx(mask, mask, cv::MORPH_OPEN, kernel);
        
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        
        cv::Mat final_small = cv::Mat::zeros(sh, sw, CV_8U);
        double max_area = 0;
        int best_idx = -1;
        
        for (int i = 0; i < contours.size(); ++i) {
            cv::Rect br = cv::boundingRect(contours[i]);
            if (br.y + br.height >= sh * 0.8) {
                double area = cv::contourArea(contours[i]);
                if (area > max_area) {
                    max_area = area;
                    best_idx = i;
                }
            }
        }
        
        if (best_idx != -1) {
            cv::drawContours(final_small, contours, best_idx, 255, -1);
            cv::Mat full_mask;
            cv::resize(final_small, full_mask, cv::Size(w, h), 0, 0, cv::INTER_LINEAR);
            
            // Convert back to numpy
            py::array_t<uint8_t> result({h, w});
            auto res_ref = result.mutable_unchecked<2>();
            std::memcpy(res_ref.mutable_data(0, 0), full_mask.data, h * w);
            
            // Calculate center of mass for path center
            cv::Moments m = cv::moments(contours[best_idx]);
            double cx = (m.m00 != 0) ? (m.m10 / m.m00) / scale : w/2;
            
            return py::make_tuple(result, cx);
        }
        
        return py::none();
    }
};

PYBIND11_MODULE(astra_vision, m) {
    py::class_<VisionAccelerator>(m, "VisionAccelerator")
        .def(py::init<>())
        .def("process_drivable_area", &VisionAccelerator::process_drivable_area);
}
