#include <algorithm>
#include <chrono>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <iomanip>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "std_msgs/msg/bool.hpp"
#include "vision_msgs/msg/bounding_box2_d.hpp"
#include "vision_msgs/msg/detection2_d.hpp"
#include "vision_msgs/msg/detection2_d_array.hpp"
#include "vision_msgs/msg/object_hypothesis_with_pose.hpp"

using namespace std::chrono_literals;

namespace cat_detector_cpp
{

// YOLOv8 ONNX export output layout: (1, 4 + num_classes, num_anchors).
// No objectness score (unlike v5) -- columns 4.. are per-class scores directly.
constexpr int kBoxCols = 4;

class DetectorNode : public rclcpp::Node
{
public:
  DetectorNode()
  : rclcpp::Node("cat_detector_cpp")
  {
    model_path_ = declare_parameter<std::string>(
      "model_path",
      "/home/jetson/yahboomcar_ros2_ws/yahboomcar_ws/src/cat_detector_cpp/models/yolov8n.onnx");
    use_cuda_ = declare_parameter<bool>("use_cuda", true);
    imgsz_ = declare_parameter<int>("imgsz", 640);
    conf_threshold_ = declare_parameter<double>("conf_threshold", 0.45);
    iou_threshold_ = declare_parameter<double>("iou_threshold", 0.50);
    const auto target_classes_i64 = declare_parameter<std::vector<int64_t>>(
      "target_classes", std::vector<int64_t>{15});
    target_classes_.assign(target_classes_i64.begin(), target_classes_i64.end());

    image_topic_ = declare_parameter<std::string>("image_topic", "/camera/color/image_raw");
    detections_topic_ = declare_parameter<std::string>(
      "detections_topic", "/cat_detector_cpp/detections");
    annotated_topic_ = declare_parameter<std::string>(
      "annotated_topic", "/cat_detector_cpp/image_annotated");
    publish_annotated_ = declare_parameter<bool>("publish_annotated", true);
    log_fps_every_sec_ = declare_parameter<double>("log_fps_every_sec", 5.0);
    tick_period_ms_ = declare_parameter<int>("tick_period_ms", 5);

    // --- Persistent dataset capture on sighting (Phase 5b flywheel) ---
    save_on_detection_ = declare_parameter<bool>("save_on_detection", true);
    detection_image_dir_ = declare_parameter<std::string>(
      "detection_image_dir", "/home/jetson/cat_patrol_data/detections_cpp");
    sighting_gap_sec_ = declare_parameter<double>("sighting_gap_sec", 3.0);
    detection_save_interval_sec_ = declare_parameter<double>("detection_save_interval_sec", 5.0);
    max_detection_images_ = declare_parameter<int>("max_detection_images", 500);
    std::filesystem::create_directories(detection_image_dir_);

    // --- Beep on sighting ---
    beep_on_detection_ = declare_parameter<bool>("beep_on_detection", true);
    const auto beep_topic = declare_parameter<std::string>("beep_topic", "Buzzer");
    beep_duration_sec_ = declare_parameter<double>("beep_duration_sec", 0.15);

    RCLCPP_INFO(get_logger(), "Loading ONNX model %s (use_cuda=%d)",
                model_path_.c_str(), use_cuda_);
    net_ = cv::dnn::readNetFromONNX(model_path_);
    if (use_cuda_) {
      net_.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
      net_.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
    } else {
      net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
      net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    }

    det_pub_ = create_publisher<vision_msgs::msg::Detection2DArray>(detections_topic_, 10);
    if (publish_annotated_) {
      ann_pub_ = create_publisher<sensor_msgs::msg::Image>(annotated_topic_, 10);
    }
    if (beep_on_detection_) {
      beep_pub_ = create_publisher<std_msgs::msg::Bool>(beep_topic, 10);
    }

    image_sub_ = create_subscription<sensor_msgs::msg::Image>(
      image_topic_, rclcpp::SensorDataQoS(),
      [this](sensor_msgs::msg::Image::SharedPtr msg) {
        // Latest-wins: only ever hold the newest frame. If inference is
        // slower than the camera, intermediate frames are simply
        // overwritten here -- dropped stale, bounded memory. Single-threaded
        // executor (see main()), so no lock needed: this callback and tick()
        // never run concurrently.
        latest_msg_ = msg;
      });

    tick_timer_ = create_wall_timer(
      std::chrono::milliseconds(tick_period_ms_), std::bind(&DetectorNode::tick, this));

    frames_ = 0;
    t0_ = now();

    RCLCPP_INFO(get_logger(), "cat_detector_cpp started. image_topic=%s detections_topic=%s "
                "detection_image_dir=%s", image_topic_.c_str(), detections_topic_.c_str(),
                detection_image_dir_.c_str());
  }

private:
  static bool image_to_bgr(const sensor_msgs::msg::Image & msg, cv::Mat & out)
  {
    const int rows = static_cast<int>(msg.height);
    const int cols = static_cast<int>(msg.width);
    const int step = static_cast<int>(msg.step);
    if (msg.encoding == "rgb8") {
      cv::Mat rgb(rows, cols, CV_8UC3, const_cast<uint8_t *>(msg.data.data()), step);
      cv::cvtColor(rgb, out, cv::COLOR_RGB2BGR);
    } else if (msg.encoding == "bgr8") {
      cv::Mat src(rows, cols, CV_8UC3, const_cast<uint8_t *>(msg.data.data()), step);
      out = src.clone();
    } else if (msg.encoding == "mono8") {
      cv::Mat mono(rows, cols, CV_8UC1, const_cast<uint8_t *>(msg.data.data()), step);
      cv::cvtColor(mono, out, cv::COLOR_GRAY2BGR);
    } else {
      return false;
    }
    return true;
  }

  static sensor_msgs::msg::Image bgr_to_image_msg(
    const cv::Mat & bgr, const std_msgs::msg::Header & header)
  {
    sensor_msgs::msg::Image msg;
    msg.header = header;
    msg.height = bgr.rows;
    msg.width = bgr.cols;
    msg.encoding = "bgr8";
    msg.is_bigendian = false;
    msg.step = static_cast<uint32_t>(bgr.cols * bgr.elemSize());
    const size_t size = msg.step * bgr.rows;
    msg.data.resize(size);
    if (bgr.isContinuous()) {
      std::memcpy(msg.data.data(), bgr.data, size);
    } else {
      for (int r = 0; r < bgr.rows; ++r) {
        std::memcpy(msg.data.data() + r * msg.step, bgr.ptr(r), msg.step);
      }
    }
    return msg;
  }

  void tick()
  {
    check_buzzer_off();

    if (!latest_msg_) {
      return;
    }
    auto msg = latest_msg_;
    latest_msg_.reset();  // consume

    cv::Mat frame;
    if (!image_to_bgr(*msg, frame)) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 10000,
        "Unsupported encoding '%s'", msg->encoding.c_str());
      return;
    }

    const auto detections = infer(frame);

    vision_msgs::msg::Detection2DArray out;
    out.header = msg->header;
    for (const auto & d : detections) {
      vision_msgs::msg::Detection2D det;
      det.header = msg->header;
      det.bbox.center.position.x = d.x + d.w / 2.0;
      det.bbox.center.position.y = d.y + d.h / 2.0;
      det.bbox.size_x = d.w;
      det.bbox.size_y = d.h;
      vision_msgs::msg::ObjectHypothesisWithPose hyp;
      hyp.hypothesis.class_id = std::to_string(d.class_id);
      hyp.hypothesis.score = d.score;
      det.results.push_back(hyp);
      out.detections.push_back(det);
    }
    det_pub_->publish(out);

    if (!out.detections.empty()) {
      handle_sighting(frame);
    }

    if (publish_annotated_) {
      cv::Mat annotated = frame.clone();
      for (const auto & d : detections) {
        cv::rectangle(annotated, cv::Rect(d.x, d.y, d.w, d.h), cv::Scalar(0, 255, 0), 2);
        std::ostringstream label;
        label << "cat " << static_cast<int>(d.score * 100) << "%";
        cv::putText(annotated, label.str(), cv::Point(d.x, std::max(0, d.y - 5)),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
      }
      ann_pub_->publish(bgr_to_image_msg(annotated, msg->header));
    }

    ++frames_;
    const auto t_now = now();
    if ((t_now - t0_).seconds() >= log_fps_every_sec_) {
      const double fps = frames_ / (t_now - t0_).seconds();
      RCLCPP_INFO(get_logger(), "inference %.1f FPS, %zu det", fps, out.detections.size());
      frames_ = 0;
      t0_ = t_now;
    }
  }

  struct Detection
  {
    int x, y, w, h;
    int class_id;
    float score;
  };

  std::vector<Detection> infer(const cv::Mat & frame)
  {
    cv::Mat blob = cv::dnn::blobFromImage(
      frame, 1.0 / 255.0, cv::Size(imgsz_, imgsz_), cv::Scalar(), true, false);
    net_.setInput(blob);
    cv::Mat output = net_.forward();  // (1, 4+num_classes, num_anchors)

    const int dims = output.size[1];        // 4 + num_classes
    const int num_anchors = output.size[2]; // e.g. 8400 for 640x640 input
    const int num_classes = dims - kBoxCols;
    // output's memory is (channel-major, anchor-minor): a flat view as a
    // (dims x num_anchors) matrix is exactly its native layout, no copy
    // needed for that step. Transpose so each row is one candidate instead:
    // [cx, cy, w, h, class0..classN].
    cv::Mat candidates_raw(dims, num_anchors, CV_32F, output.ptr<float>());
    cv::Mat candidates = candidates_raw.t();

    const double scale_x = frame.cols / static_cast<double>(imgsz_);
    const double scale_y = frame.rows / static_cast<double>(imgsz_);

    std::vector<cv::Rect> boxes;
    std::vector<float> scores;
    std::vector<int> class_ids;

    for (int i = 0; i < candidates.rows; ++i) {
      const cv::Mat class_scores = candidates.row(i).colRange(kBoxCols, kBoxCols + num_classes);
      cv::Point max_loc;
      double max_score;
      cv::minMaxLoc(class_scores, nullptr, &max_score, nullptr, &max_loc);
      const int class_id = max_loc.x;
      if (max_score < conf_threshold_) {
        continue;
      }
      if (std::find(target_classes_.begin(), target_classes_.end(), class_id) ==
          target_classes_.end())
      {
        continue;
      }

      const float cx = candidates.at<float>(i, 0);
      const float cy = candidates.at<float>(i, 1);
      const float w = candidates.at<float>(i, 2);
      const float h = candidates.at<float>(i, 3);

      const int x1 = static_cast<int>((cx - w / 2.0) * scale_x);
      const int y1 = static_cast<int>((cy - h / 2.0) * scale_y);
      const int bw = static_cast<int>(w * scale_x);
      const int bh = static_cast<int>(h * scale_y);

      boxes.emplace_back(x1, y1, bw, bh);
      scores.push_back(static_cast<float>(max_score));
      class_ids.push_back(class_id);
    }

    std::vector<int> keep;
    cv::dnn::NMSBoxes(boxes, scores, static_cast<float>(conf_threshold_),
                       static_cast<float>(iou_threshold_), keep);

    std::vector<Detection> out;
    out.reserve(keep.size());
    for (int idx : keep) {
      out.push_back(Detection{
        boxes[idx].x, boxes[idx].y, boxes[idx].width, boxes[idx].height,
        class_ids[idx], scores[idx]});
    }
    return out;
  }

  // Called once per frame that contains >=1 target-class detection. Beeps
  // once per NEW sighting (a gap of sighting_gap_sec_ with no cat counts as
  // "new"), and saves a timestamped frame for the Phase 5b dataset --
  // immediately on a new sighting, then throttled to at most one more every
  // detection_save_interval_sec_ while it continues.
  void handle_sighting(const cv::Mat & frame)
  {
    const auto t_now = std::chrono::steady_clock::now();
    const bool is_new_sighting =
      !last_seen_time_.has_value() ||
      std::chrono::duration<double>(t_now - *last_seen_time_).count() > sighting_gap_sec_;
    last_seen_time_ = t_now;

    if (is_new_sighting) {
      RCLCPP_INFO(get_logger(), "Cat sighting started");
      if (beep_on_detection_) {
        beep();
      }
      if (save_on_detection_) {
        save_frame(frame);
        last_saved_time_ = t_now;
      }
    } else if (save_on_detection_ &&
      (!last_saved_time_.has_value() ||
       std::chrono::duration<double>(t_now - *last_saved_time_).count() >=
         detection_save_interval_sec_))
    {
      save_frame(frame);
      last_saved_time_ = t_now;
    }
  }

  void beep()
  {
    std_msgs::msg::Bool on;
    on.data = true;
    beep_pub_->publish(on);
    buzzer_on_ = true;
    buzzer_off_time_ = now() + rclcpp::Duration::from_seconds(beep_duration_sec_);
  }

  void check_buzzer_off()
  {
    if (buzzer_on_ && now() >= buzzer_off_time_) {
      std_msgs::msg::Bool off;
      off.data = false;
      beep_pub_->publish(off);
      buzzer_on_ = false;
    }
  }

  void save_frame(const cv::Mat & frame)
  {
    const auto t = std::chrono::system_clock::now();
    const auto t_sec = std::chrono::system_clock::to_time_t(t);
    const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
      t.time_since_epoch()) % 1000;

    std::tm tm_buf{};
    localtime_r(&t_sec, &tm_buf);
    std::ostringstream ts;
    ts << std::put_time(&tm_buf, "%Y%m%d_%H%M%S") << "_"
       << std::setw(3) << std::setfill('0') << ms.count();

    const auto fp = (std::filesystem::path(detection_image_dir_) / ("cat_" + ts.str() + ".jpg")).string();
    if (cv::imwrite(fp, frame)) {
      RCLCPP_INFO(get_logger(), "Saved detection frame: %s", fp.c_str());
      enforce_retention();
    }
  }

  void enforce_retention()
  {
    std::vector<std::filesystem::path> files;
    for (const auto & entry : std::filesystem::directory_iterator(detection_image_dir_)) {
      const auto name = entry.path().filename().string();
      if (name.rfind("cat_", 0) == 0 && entry.path().extension() == ".jpg") {
        files.push_back(entry.path());
      }
    }
    // Filenames are cat_<timestamp>.jpg, so a plain path sort is also a
    // chronological sort -- oldest files come first.
    std::sort(files.begin(), files.end());
    const int excess = static_cast<int>(files.size()) - max_detection_images_;
    for (int i = 0; i < excess; ++i) {
      std::error_code ec;
      std::filesystem::remove(files[i], ec);
      if (ec) {
        RCLCPP_WARN(get_logger(), "Failed to evict %s: %s",
                    files[i].c_str(), ec.message().c_str());
      }
    }
  }

  // --- Parameters ---
  std::string model_path_;
  bool use_cuda_{true};
  int imgsz_{640};
  double conf_threshold_{0.45};
  double iou_threshold_{0.50};
  std::vector<int> target_classes_;
  std::string image_topic_;
  std::string detections_topic_;
  std::string annotated_topic_;
  bool publish_annotated_{true};
  double log_fps_every_sec_{5.0};
  int tick_period_ms_{5};

  bool save_on_detection_{true};
  std::string detection_image_dir_;
  double sighting_gap_sec_{3.0};
  double detection_save_interval_sec_{5.0};
  int max_detection_images_{500};

  bool beep_on_detection_{true};
  double beep_duration_sec_{0.15};

  // --- State ---
  sensor_msgs::msg::Image::SharedPtr latest_msg_;
  std::optional<std::chrono::steady_clock::time_point> last_seen_time_;
  std::optional<std::chrono::steady_clock::time_point> last_saved_time_;
  bool buzzer_on_{false};
  rclcpp::Time buzzer_off_time_;
  int frames_{0};
  rclcpp::Time t0_;

  // --- ROS entities ---
  cv::dnn::Net net_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
  rclcpp::Publisher<vision_msgs::msg::Detection2DArray>::SharedPtr det_pub_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr ann_pub_;
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr beep_pub_;
  rclcpp::TimerBase::SharedPtr tick_timer_;
};

}  // namespace cat_detector_cpp

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<cat_detector_cpp::DetectorNode>());
  rclcpp::shutdown();
  return 0;
}
