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

// Must match collect_dataset.py's PAD_FRAC / the Python detector_node.py's
// ID_CROP_PAD_FRAC -- the classifier was trained on crops padded this much
// around the YOLO box, so live crops need the same padding or accuracy
// silently degrades.
constexpr double kIdCropPadFrac = 0.15;
constexpr int kIdResizeShortSide = 256;  // matches transforms.Resize(256)
constexpr int kIdCropSize = 224;          // matches transforms.CenterCrop(224)

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

    // --- Per-cat identification (Phase 5b) ---
    enable_identification_ = declare_parameter<bool>("enable_identification", true);
    classifier_model_path_ = declare_parameter<std::string>(
      "classifier_model_path",
      "/home/jetson/yahboomcar_ros2_ws/yahboomcar_ws/src/cat_detector_cpp/models/cat_id_classifier.onnx");
    classifier_classes_ = declare_parameter<std::vector<std::string>>(
      "classifier_classes", std::vector<std::string>{"brown", "white"});

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

    if (enable_identification_) {
      if (!std::filesystem::exists(classifier_model_path_)) {
        RCLCPP_WARN(get_logger(),
          "enable_identification=true but no classifier at %s -- falling "
          "back to generic 'cat' detection only (run export_classifier_onnx.py first).",
          classifier_model_path_.c_str());
        enable_identification_ = false;
      } else {
        RCLCPP_INFO(get_logger(), "Loading classifier %s (classes: %zu)",
                    classifier_model_path_.c_str(), classifier_classes_.size());
        classifier_net_ = cv::dnn::readNetFromONNX(classifier_model_path_);
        if (use_cuda_) {
          classifier_net_.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
          classifier_net_.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
        } else {
          classifier_net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
          classifier_net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
        }
      }
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

    // Parallel to `detections`: per-box (label, confidence) from the
    // per-cat classifier, or ("", -1) if identification didn't fire.
    std::vector<std::pair<std::string, float>> identities;
    identities.reserve(detections.size());
    for (const auto & d : detections) {
      identities.push_back(classify_crop(frame, cv::Rect(d.x, d.y, d.w, d.h)));
    }

    vision_msgs::msg::Detection2DArray out;
    out.header = msg->header;
    for (size_t i = 0; i < detections.size(); ++i) {
      const auto & d = detections[i];
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

      const auto & [id_label, id_conf] = identities[i];
      if (!id_label.empty()) {
        vision_msgs::msg::ObjectHypothesisWithPose id_hyp;
        id_hyp.hypothesis.class_id = id_label;
        id_hyp.hypothesis.score = id_conf;
        det.results.push_back(id_hyp);
      }

      out.detections.push_back(det);
    }
    det_pub_->publish(out);

    if (!out.detections.empty()) {
      handle_sighting(frame, identities);
    }

    if (publish_annotated_) {
      cv::Mat annotated = frame.clone();
      for (size_t i = 0; i < detections.size(); ++i) {
        const auto & d = detections[i];
        cv::rectangle(annotated, cv::Rect(d.x, d.y, d.w, d.h), cv::Scalar(0, 255, 0), 2);
        std::ostringstream label;
        const auto & [id_label, id_conf] = identities[i];
        if (!id_label.empty()) {
          label << id_label << " " << static_cast<int>(id_conf * 100) << "%";
        } else {
          label << "cat " << static_cast<int>(d.score * 100) << "%";
        }
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

  // Crop the detection box out of the full frame, padded by kIdCropPadFrac
  // (matching the training-time crop padding), clamped to the frame bounds.
  static cv::Mat crop_for_identification(const cv::Mat & frame, const cv::Rect & box)
  {
    const int w = frame.cols;
    const int h = frame.rows;
    const double px = box.width * kIdCropPadFrac;
    const double py = box.height * kIdCropPadFrac;
    const int x1 = std::max(0, static_cast<int>(box.x - px));
    const int y1 = std::max(0, static_cast<int>(box.y - py));
    const int x2 = std::min(w, static_cast<int>(box.x + box.width + px));
    const int y2 = std::min(h, static_cast<int>(box.y + box.height + py));
    if (x2 <= x1 || y2 <= y1) {
      return cv::Mat();
    }
    return frame(cv::Rect(x1, y1, x2 - x1, y2 - y1)).clone();
  }

  // Reproduces train_classifier.py's eval-time preprocessing
  // (Resize(256) + CenterCrop(224) + ToTensor + ImageNet Normalize) using
  // plain OpenCV ops rather than cv::dnn::blobFromImage's built-in mean/scale,
  // because blobFromImage only supports a single uniform scalefactor across
  // channels and ImageNet normalization needs a different scale factor
  // (1/std) per channel. INTER_AREA is used for the downscale specifically
  // because it was verified (bench comparison against the PyTorch/PIL
  // reference) to track PIL's antialiased Resize much more closely than
  // INTER_LINEAR -- the two backends are never bit-exact, but this keeps
  // predicted-class agreement on every example tried.
  static cv::Mat preprocess_for_classifier(const cv::Mat & bgr_crop)
  {
    const int h = bgr_crop.rows;
    const int w = bgr_crop.cols;
    const double scale = kIdResizeShortSide / static_cast<double>(std::min(h, w));
    const int new_w = static_cast<int>(std::round(w * scale));
    const int new_h = static_cast<int>(std::round(h * scale));

    cv::Mat resized;
    cv::resize(bgr_crop, resized, cv::Size(new_w, new_h), 0, 0, cv::INTER_AREA);

    const int x0 = std::clamp((new_w - kIdCropSize) / 2, 0, std::max(0, new_w - kIdCropSize));
    const int y0 = std::clamp((new_h - kIdCropSize) / 2, 0, std::max(0, new_h - kIdCropSize));
    cv::Mat cropped = resized(cv::Rect(x0, y0, kIdCropSize, kIdCropSize));

    cv::Mat rgb;
    cv::cvtColor(cropped, rgb, cv::COLOR_BGR2RGB);
    cv::Mat float_img;
    rgb.convertTo(float_img, CV_32FC3, 1.0 / 255.0);

    static const cv::Scalar kMean(0.485, 0.456, 0.406);
    static const cv::Scalar kStd(0.229, 0.224, 0.225);
    std::vector<cv::Mat> channels(3);
    cv::split(float_img, channels);
    for (int c = 0; c < 3; ++c) {
      channels[c] = (channels[c] - kMean[c]) / kStd[c];
    }
    cv::Mat normalized;
    cv::merge(channels, normalized);

    // Already normalized -> blobFromImage just does HWC->CHW + batch dim.
    return cv::dnn::blobFromImage(normalized);
  }

  // Returns (label, confidence), or ("", -1) if identification is disabled,
  // the classifier isn't loaded, or the padded box crops to nothing.
  std::pair<std::string, float> classify_crop(const cv::Mat & frame, const cv::Rect & box)
  {
    if (!enable_identification_) {
      return {"", -1.0f};
    }
    cv::Mat crop = crop_for_identification(frame, box);
    if (crop.empty()) {
      return {"", -1.0f};
    }

    cv::Mat blob = preprocess_for_classifier(crop);
    classifier_net_.setInput(blob);
    cv::Mat output = classifier_net_.forward();  // (1, num_classes) raw logits

    cv::Mat flat = output.reshape(1, 1);
    double max_logit;
    cv::minMaxLoc(flat, nullptr, &max_logit);
    cv::Mat exp_scores;
    cv::exp(flat - max_logit, exp_scores);
    const double sum = cv::sum(exp_scores)[0];
    exp_scores /= sum;  // softmax

    cv::Point max_loc;
    double max_prob;
    cv::minMaxLoc(exp_scores, nullptr, &max_prob, nullptr, &max_loc);
    const int idx = max_loc.x;
    if (idx < 0 || idx >= static_cast<int>(classifier_classes_.size())) {
      return {"", -1.0f};
    }
    return {classifier_classes_[idx], static_cast<float>(max_prob)};
  }

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
  void handle_sighting(
    const cv::Mat & frame,
    const std::vector<std::pair<std::string, float>> & identities)
  {
    const auto t_now = std::chrono::steady_clock::now();
    const bool is_new_sighting =
      !last_seen_time_.has_value() ||
      std::chrono::duration<double>(t_now - *last_seen_time_).count() > sighting_gap_sec_;
    last_seen_time_ = t_now;

    // Best identified label across all boxes in this frame (highest
    // classifier confidence), matching detector_node.py's approach.
    std::string label;
    float best_conf = -1.0f;
    for (const auto & [id_label, id_conf] : identities) {
      if (!id_label.empty() && id_conf > best_conf) {
        label = id_label;
        best_conf = id_conf;
      }
    }

    if (is_new_sighting) {
      RCLCPP_INFO(get_logger(), "Cat sighting started%s",
                  label.empty() ? "" : (" (" + label + ")").c_str());
      if (beep_on_detection_) {
        beep();
      }
      if (save_on_detection_) {
        save_frame(frame, label);
        last_saved_time_ = t_now;
      }
    } else if (save_on_detection_ &&
      (!last_saved_time_.has_value() ||
       std::chrono::duration<double>(t_now - *last_saved_time_).count() >=
         detection_save_interval_sec_))
    {
      save_frame(frame, label);
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

  void save_frame(const cv::Mat & frame, const std::string & label = "")
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

    // Label suffix comes after the timestamp so cat_*.jpg sorting/eviction
    // in enforce_retention (chronological via plain path sort) still works.
    const std::string filename = label.empty()
      ? "cat_" + ts.str() + ".jpg"
      : "cat_" + ts.str() + "_" + label + ".jpg";
    const auto fp = (std::filesystem::path(detection_image_dir_) / filename).string();
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

  bool enable_identification_{true};
  std::string classifier_model_path_;
  std::vector<std::string> classifier_classes_;

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
  cv::dnn::Net classifier_net_;
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
