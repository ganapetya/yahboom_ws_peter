#include <chrono>
#include <cmath>
#include <filesystem>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include <opencv2/opencv.hpp>

#include "geometry_msgs/msg/pose_stamped.hpp"
#include "geometry_msgs/msg/twist.hpp"
#include "nav2_msgs/action/navigate_to_pose.hpp"
#include "rclcpp/rclcpp.hpp"
#include "rclcpp_action/rclcpp_action.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "std_msgs/msg/string.hpp"
#include "vision_msgs/msg/detection2_d_array.hpp"
#include "visualization_msgs/msg/marker.hpp"
#include "visualization_msgs/msg/marker_array.hpp"

#include "patrol_manager/waypoint_patrol_strategy.hpp"

using NavigateToPose = nav2_msgs::action::NavigateToPose;
using GoalHandle = rclcpp_action::ClientGoalHandle<NavigateToPose>;
using namespace std::chrono_literals;

namespace patrol_manager
{

enum class State
{
  SLEEPING,
  WAKING,
  PATROLLING,
  CAPTURE,
  RETURNING,
  INVESTIGATING,
};

class PatrolManagerNode : public rclcpp::Node
{
public:
  PatrolManagerNode()
  : rclcpp::Node("patrol_manager")
  {
    // Parameters
    start_on_boot_ = declare_parameter<bool>("start_on_boot", true);
    loop_patrol_ = declare_parameter<bool>("loop_patrol", true);
    patrol_period_sec_ = declare_parameter<double>("patrol_period_sec", 1800.0);
    patrol_pause_sec_ = declare_parameter<double>("patrol_pause_sec", 10.0);
    settle_after_arrival_sec_ = declare_parameter<double>("settle_after_arrival_sec", 1.0);

    image_topic_ = declare_parameter<std::string>("image_topic", "/camera/color/image_raw");
    mail_request_topic_ = declare_parameter<std::string>("mail_request_topic", "/cat_patrol/mail_request");
    state_topic_ = declare_parameter<std::string>("state_topic", "/patrol_manager/state");
    image_save_dir_ = declare_parameter<std::string>("image_save_dir", "/tmp/cat_patrol_images");
    mail_subject_ = declare_parameter<std::string>("mail_subject", "Cat patrol photos");
    mail_to_ = declare_parameter<std::string>("smtp_to_address", "user@example.com");
    map_frame_ = declare_parameter<std::string>("map_frame", "map");

    // --- Cat-recognized snapshot email (fires only during an active cycle) ---
    // cat_detector_cpp publishes a Detection2DArray with the white/brown identity
    // in results[1]; detections_cb() emails a snapshot on a confident sighting.
    detections_topic_ = declare_parameter<std::string>(
      "detections_topic", "/cat_detector_cpp/detections");
    cat_recognized_mail_enabled_ = declare_parameter<bool>("cat_recognized_mail_enabled", true);
    cat_recognized_min_conf_ = declare_parameter<double>("cat_recognized_min_conf", 0.6);
    cat_recognized_cooldown_sec_ =
      declare_parameter<double>("cat_recognized_cooldown_sec", 30.0);
    // --- Loop-boundary snapshot emails (start of WAKING, end at home) ---
    loop_boundary_mail_enabled_ = declare_parameter<bool>("loop_boundary_mail_enabled", true);

    // Stall guard: compares commanded velocity (cmd_vel_topic) against actual
    // wheel-encoder velocity (vel_raw_topic). Nav2's own progress checker only
    // watches position over ~10-20s; this catches a robot that is commanded to
    // move but physically can't (jammed against an unseen obstacle) faster,
    // and works no matter which Nav2 state (goal, recovery) is commanding it.
    cmd_vel_topic_ = declare_parameter<std::string>("cmd_vel_topic", "/cmd_vel");
    vel_raw_topic_ = declare_parameter<std::string>("vel_raw_topic", "/vel_raw");
    stall_timeout_sec_ = declare_parameter<double>("stall_timeout_sec", 6.0);
    stall_cmd_vel_threshold_ = declare_parameter<double>("stall_cmd_vel_threshold", 0.03);
    stall_vel_raw_threshold_ = declare_parameter<double>("stall_vel_raw_threshold", 0.02);

    const auto home_pose = declare_parameter<std::vector<double>>("home_pose", std::vector<double>{0.0, 0.0, 0.0});
    if (home_pose.size() == 3) {
      home_.x = home_pose[0];
      home_.y = home_pose[1];
      home_.yaw = home_pose[2];
      home_.name = "home";
    }

    const auto names = declare_parameter<std::vector<std::string>>("waypoint_names", std::vector<std::string>{});
    const auto xs = declare_parameter<std::vector<double>>("waypoint_x", std::vector<double>{});
    const auto ys = declare_parameter<std::vector<double>>("waypoint_y", std::vector<double>{});
    const auto yaws = declare_parameter<std::vector<double>>("waypoint_yaw", std::vector<double>{});

    if (!(names.size() == xs.size() && xs.size() == ys.size() && ys.size() == yaws.size())) {
      throw std::runtime_error("waypoint arrays must have equal sizes");
    }
    std::vector<Pose2D> waypoints;
    waypoints.reserve(names.size());
    for (std::size_t i = 0; i < names.size(); ++i) {
      waypoints.push_back(Pose2D{xs[i], ys[i], yaws[i], names[i]});
    }

    // Latched (transient_local) so RViz sees the markers even if added after
    // this node started. Published once here from the same data the FSM
    // navigates to, so what you see in RViz is guaranteed to match reality.
    waypoints_marker_pub_ = create_publisher<visualization_msgs::msg::MarkerArray>(
      "/patrol_manager/waypoints", rclcpp::QoS(1).transient_local());
    publish_waypoint_markers(waypoints);

    strategy_ = std::make_unique<WaypointPatrolStrategy>(std::move(waypoints));

    std::filesystem::create_directories(image_save_dir_);

    // Callback groups
    fsm_group_ = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
    io_group_ = create_callback_group(rclcpp::CallbackGroupType::Reentrant);

    // I/O
    rclcpp::SubscriptionOptions sub_opts;
    sub_opts.callback_group = io_group_;
    image_sub_ = create_subscription<sensor_msgs::msg::Image>(
      image_topic_, rclcpp::SensorDataQoS(),
      [this](sensor_msgs::msg::Image::SharedPtr msg) { last_image_ = msg; },
      sub_opts);

    cmd_vel_sub_ = create_subscription<geometry_msgs::msg::Twist>(
      cmd_vel_topic_, rclcpp::SystemDefaultsQoS(),
      [this](geometry_msgs::msg::Twist::SharedPtr msg) { last_cmd_vel_ = msg; },
      sub_opts);
    vel_raw_sub_ = create_subscription<geometry_msgs::msg::Twist>(
      vel_raw_topic_, rclcpp::SystemDefaultsQoS(),
      [this](geometry_msgs::msg::Twist::SharedPtr msg) { last_vel_raw_ = msg; },
      sub_opts);

    // Cat identity stream from cat_detector_cpp. Placed on the MutuallyExclusive
    // fsm_group_ (not io_group_) so detections_cb() is serialized with tick() and
    // the action result callbacks — it reads state_ and writes last_cat_mail_, and
    // must not race the FSM under the MultiThreadedExecutor.
    if (cat_recognized_mail_enabled_) {
      rclcpp::SubscriptionOptions det_opts;
      det_opts.callback_group = fsm_group_;
      detections_sub_ = create_subscription<vision_msgs::msg::Detection2DArray>(
        detections_topic_, 10,
        std::bind(&PatrolManagerNode::detections_cb, this, std::placeholders::_1),
        det_opts);
    }

    mail_pub_ = create_publisher<std_msgs::msg::String>(mail_request_topic_, 10);
    state_pub_ = create_publisher<std_msgs::msg::String>(state_topic_, 10);

    // NOTE: the action client shares the MutuallyExclusive fsm_group_ so that
    // goal/feedback/result callbacks are serialized with the FSM tick. Without
    // this, result callbacks land in the node default group and race the tick
    // on state_/strategy_/cycle_image_paths_ under MultiThreadedExecutor.
    action_client_ = rclcpp_action::create_client<NavigateToPose>(this, "navigate_to_pose", fsm_group_);

    // Timers
    fsm_timer_ = create_wall_timer(100ms, std::bind(&PatrolManagerNode::tick, this), fsm_group_);
    wake_timer_ = create_wall_timer(
      std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::duration<double>(patrol_period_sec_)),
      [this]() { wake_requested_ = true; },
      fsm_group_);

    state_ = start_on_boot_ ? State::WAKING : State::SLEEPING;
    publish_state();

    RCLCPP_INFO(get_logger(), "PatrolManager started. strategy=%s waypoints=%zu",
                strategy_->name().c_str(), names.size());
  }

private:
  static geometry_msgs::msg::PoseStamped to_pose(const Pose2D & p, const std::string & frame, const rclcpp::Time & stamp)
  {
    geometry_msgs::msg::PoseStamped out;
    out.header.frame_id = frame;
    out.header.stamp = stamp;
    out.pose.position.x = p.x;
    out.pose.position.y = p.y;
    out.pose.orientation.z = std::sin(p.yaw / 2.0);
    out.pose.orientation.w = std::cos(p.yaw / 2.0);
    return out;
  }

  void publish_waypoint_markers(const std::vector<Pose2D> & waypoints)
  {
    visualization_msgs::msg::MarkerArray markers;
    const auto stamp = now();
    int id = 0;

    auto add_pose = [&](const Pose2D & p, float r, float g, float b) {
      visualization_msgs::msg::Marker arrow;
      arrow.header.frame_id = map_frame_;
      arrow.header.stamp = stamp;
      arrow.ns = "patrol_waypoints";
      arrow.id = id++;
      arrow.type = visualization_msgs::msg::Marker::ARROW;
      arrow.action = visualization_msgs::msg::Marker::ADD;
      arrow.pose = to_pose(p, map_frame_, stamp).pose;
      arrow.scale.x = 0.35;  // shaft length
      arrow.scale.y = 0.06;  // shaft diameter
      arrow.scale.z = 0.06;  // head diameter
      arrow.color.r = r;
      arrow.color.g = g;
      arrow.color.b = b;
      arrow.color.a = 1.0;
      markers.markers.push_back(arrow);

      visualization_msgs::msg::Marker label;
      label.header = arrow.header;
      label.ns = "patrol_waypoint_labels";
      label.id = id++;
      label.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
      label.action = visualization_msgs::msg::Marker::ADD;
      label.pose = arrow.pose;
      label.pose.position.z += 0.25;
      label.scale.z = 0.18;  // text height
      label.color.r = r;
      label.color.g = g;
      label.color.b = b;
      label.color.a = 1.0;
      label.text = p.name;
      markers.markers.push_back(label);
    };

    for (const auto & wp : waypoints) {
      add_pose(wp, 0.1f, 0.6f, 1.0f);  // blue: patrol waypoints
    }
    add_pose(home_, 1.0f, 0.1f, 0.1f);  // red: home

    waypoints_marker_pub_->publish(markers);
    RCLCPP_INFO(get_logger(), "Published %zu waypoint markers to /patrol_manager/waypoints (frame=%s)",
                waypoints.size() + 1, map_frame_.c_str());
  }

  void tick()
  {
    check_stall();

    switch (state_) {
      case State::SLEEPING:
        if (wake_requested_) {
          wake_requested_ = false;
          transition(State::WAKING);
        }
        break;

      case State::WAKING:
        cycle_image_paths_.clear();
        strategy_->reset();
        // Non-blocking readiness check — never block inside the FSM timer (see §9).
        // Stay in WAKING and re-check next tick until the server is up.
        if (!action_client_->action_server_is_ready()) {
          RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
            "navigate_to_pose action server unavailable; staying in WAKING");
          break;
        }
        // Loop START email: one snapshot as a new cycle begins. We only reach
        // here (server ready) once per cycle before leaving WAKING for
        // PATROLLING, so this fires exactly once per loop.
        if (loop_boundary_mail_enabled_) {
          const std::string p = save_snapshot("loopstart");
          if (!p.empty()) {
            send_mail(mail_subject_ + " - patrol loop started", {p});
          }
        }
        send_next_waypoint_goal();
        break;

      case State::PATROLLING:
        // Waiting for action result callback to transition.
        break;

      case State::CAPTURE:
        if (save_current_image()) {
          send_waypoint_mail(current_waypoint_.name, cycle_image_paths_.back());
          if (strategy_->has_next()) {
            send_next_waypoint_goal();
          } else {
            transition(State::RETURNING);
            send_home_goal();
          }
        }
        break;

      case State::RETURNING:
        // Waiting for home-goal result callback.
        break;

      case State::INVESTIGATING:
        // Reserved for Phase 6.
        break;
    }
  }

  void send_next_waypoint_goal()
  {
    if (!strategy_->has_next()) {
      transition(State::RETURNING);
      send_home_goal();
      return;
    }

    const auto wp = strategy_->next();
    current_waypoint_ = wp;
    auto goal = NavigateToPose::Goal();
    goal.pose = to_pose(wp, map_frame_, now());

    transition(State::PATROLLING);
    RCLCPP_INFO(get_logger(), "Navigating to waypoint '%s' x=%.2f y=%.2f yaw=%.2f",
                wp.name.c_str(), wp.x, wp.y, wp.yaw);

    auto opts = rclcpp_action::Client<NavigateToPose>::SendGoalOptions();
    opts.goal_response_callback =
      [this](GoalHandle::SharedPtr handle) { this->active_goal_handle_ = handle; };
    opts.feedback_callback =
      [this](GoalHandle::SharedPtr,
             const std::shared_ptr<const NavigateToPose::Feedback> fb) {
        RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 1000,
          "distance remaining: %.2f m", fb->distance_remaining);
      };

    opts.result_callback =
      [this, wp](const GoalHandle::WrappedResult & result) {
        this->active_goal_handle_.reset();
        if (this->state_ == State::SLEEPING) {
          return;  // stall guard already cancelled and stopped the cycle
        }
        if (result.code == rclcpp_action::ResultCode::SUCCEEDED) {
          this->transition(State::CAPTURE);
        } else {
          RCLCPP_ERROR(this->get_logger(),
            "Waypoint '%s' goal failed (code=%d); skipping it and continuing cycle",
            wp.name.c_str(), static_cast<int>(result.code));
          this->send_next_waypoint_goal();
        }
      };

    action_client_->async_send_goal(goal, opts);
  }

  void send_home_goal()
  {
    auto goal = NavigateToPose::Goal();
    goal.pose = to_pose(home_, map_frame_, now());

    RCLCPP_INFO(get_logger(), "Returning home x=%.2f y=%.2f yaw=%.2f", home_.x, home_.y, home_.yaw);

    auto opts = rclcpp_action::Client<NavigateToPose>::SendGoalOptions();
    opts.goal_response_callback =
      [this](GoalHandle::SharedPtr handle) { this->active_goal_handle_ = handle; };
    opts.result_callback =
      [this](const GoalHandle::WrappedResult & result) {
        this->active_goal_handle_.reset();
        if (this->state_ == State::SLEEPING) {
          return;  // stall guard already cancelled and stopped the cycle
        }
        if (result.code == rclcpp_action::ResultCode::SUCCEEDED) {
          RCLCPP_INFO(get_logger(), "Cycle complete: %zu photo(s) sent, returned home",
                      cycle_image_paths_.size());
          // Loop END email: one snapshot now that we're back home.
          if (loop_boundary_mail_enabled_) {
            const std::string p = save_snapshot("loopend");
            if (!p.empty()) {
              send_mail(mail_subject_ + " - patrol loop ended", {p});
            }
          }
          transition(State::SLEEPING);
          if (loop_patrol_) {
            RCLCPP_INFO(get_logger(), "loop_patrol=true, next cycle in %.0fs", patrol_pause_sec_);
            schedule_next_cycle();
          } else {
            RCLCPP_INFO(get_logger(), "loop_patrol=false, cycle complete");
          }
        } else {
          RCLCPP_ERROR(get_logger(), "Home goal failed (code=%d)", static_cast<int>(result.code));
          transition(State::SLEEPING);
        }
      };

    action_client_->async_send_goal(goal, opts);
  }

  void schedule_next_cycle()
  {
    // One-shot: cancel itself on first fire so cycles don't stack if the node
    // stays up a long time between wakes.
    pause_timer_ = create_wall_timer(
      std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::duration<double>(patrol_pause_sec_)),
      [this]() {
        wake_requested_ = true;
        pause_timer_->cancel();
      },
      fsm_group_);
  }

  // Manual sensor_msgs::Image -> BGR cv::Mat conversion (same approach as
  // patrol_node.cpp). Avoids a cv_bridge dependency/ABI risk on this build.
  bool decode_last_image(cv::Mat & out)
  {
    if (!last_image_ || last_image_->data.empty()) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "No image available yet");
      return false;
    }
    try {
      const std::string & enc = last_image_->encoding;
      const int rows = static_cast<int>(last_image_->height);
      const int cols = static_cast<int>(last_image_->width);
      const int step = static_cast<int>(last_image_->step);
      cv::Mat bgr_image;
      if (enc == "rgb8") {
        cv::Mat rgb(rows, cols, CV_8UC3, const_cast<uint8_t*>(last_image_->data.data()), step);
        cv::cvtColor(rgb, bgr_image, cv::COLOR_RGB2BGR);
      } else if (enc == "bgr8") {
        cv::Mat src(rows, cols, CV_8UC3, const_cast<uint8_t*>(last_image_->data.data()), step);
        bgr_image = src.clone();
      } else if (enc == "mono8") {
        cv::Mat mono(rows, cols, CV_8UC1, const_cast<uint8_t*>(last_image_->data.data()), step);
        cv::cvtColor(mono, bgr_image, cv::COLOR_GRAY2BGR);
      } else {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 10000,
          "Unsupported encoding '%s'", enc.c_str());
        return false;
      }
      if (bgr_image.empty()) {
        return false;
      }
      out = bgr_image;
      return true;
    } catch (const std::exception & e) {
      RCLCPP_WARN(get_logger(), "decode_last_image exception: %s", e.what());
      return false;
    }
  }

  // Save the current frame as a standalone "<tag>_<ts>.jpg" and return its
  // path (or "" on failure). Does NOT touch cycle_image_paths_ — used for the
  // loop-boundary and cat-recognized emails, which must not inflate the
  // per-cycle capture count reported on return home.
  std::string save_snapshot(const std::string & tag)
  {
    cv::Mat bgr_image;
    if (!decode_last_image(bgr_image)) {
      return "";
    }
    std::ostringstream fn;
    fn << tag << "_" << now().nanoseconds() << ".jpg";
    const auto fp = (std::filesystem::path(image_save_dir_) / fn.str()).string();
    if (cv::imwrite(fp, bgr_image)) {
      RCLCPP_INFO(get_logger(), "Snapshot: %s", fp.c_str());
      return fp;
    }
    RCLCPP_WARN(get_logger(), "save_snapshot: imwrite failed for %s", fp.c_str());
    return "";
  }

  // Save the current frame as a waypoint capture: appends to cycle_image_paths_
  // (the per-cycle batch) as before.
  bool save_current_image()
  {
    cv::Mat bgr_image;
    if (!decode_last_image(bgr_image)) {
      return false;
    }
    std::ostringstream fn;
    fn << "snap_" << now().nanoseconds() << ".jpg";
    const auto fp = (std::filesystem::path(image_save_dir_) / fn.str()).string();
    if (cv::imwrite(fp, bgr_image)) {
      cycle_image_paths_.push_back(fp);
      RCLCPP_INFO(get_logger(), "Captured: %s", fp.c_str());
      return true;
    }
    return false;
  }

  // True whenever a patrol cycle is running (anything but SLEEPING). Gates the
  // cat-recognized email so it never fires while parked at home between cycles.
  bool is_patrolling() const { return state_ != State::SLEEPING; }

  // Emails one snapshot when cat_detector_cpp recognizes a white/brown cat
  // mid-cycle. Mirrors voice_node.py's identity logic (results[1]); highest-
  // confidence identity in the frame, gated by min-conf and a per-cat cooldown.
  void detections_cb(const vision_msgs::msg::Detection2DArray::SharedPtr msg)
  {
    if (!cat_recognized_mail_enabled_ || !is_patrolling()) {
      return;
    }

    std::string best_label;
    double best_conf = -1.0;
    for (const auto & det : msg->detections) {
      if (det.results.size() < 2) {
        continue;  // per-cat classifier didn't fire for this box
      }
      const auto & id_hyp = det.results[1].hypothesis;
      if ((id_hyp.class_id == "white" || id_hyp.class_id == "brown") &&
          id_hyp.score > best_conf)
      {
        best_label = id_hyp.class_id;
        best_conf = id_hyp.score;
      }
    }

    if (best_label.empty() || best_conf < cat_recognized_min_conf_) {
      return;
    }

    const rclcpp::Time tnow = now();
    const auto it = last_cat_mail_.find(best_label);
    if (it != last_cat_mail_.end() &&
        (tnow - it->second).seconds() < cat_recognized_cooldown_sec_)
    {
      return;
    }

    const std::string path = save_snapshot("cat_" + best_label);
    if (path.empty()) {
      return;  // no frame yet; try again on the next detection
    }
    last_cat_mail_[best_label] = tnow;
    RCLCPP_INFO(get_logger(), "Recognized %s cat (%.2f) - emailing snapshot",
                best_label.c_str(), best_conf);
    send_mail(mail_subject_ + " - " + best_label + " cat recognized", {path});
  }

  void check_stall()
  {
    if (state_ != State::PATROLLING && state_ != State::RETURNING) {
      stall_since_.reset();
      return;
    }
    if (!last_cmd_vel_ || !last_vel_raw_) {
      return;  // no data yet
    }

    const double cmd_speed = std::hypot(last_cmd_vel_->linear.x, last_cmd_vel_->linear.y) +
      std::abs(last_cmd_vel_->angular.z);
    const double actual_speed = std::hypot(last_vel_raw_->linear.x, last_vel_raw_->linear.y) +
      std::abs(last_vel_raw_->angular.z);

    const bool commanding_motion = cmd_speed > stall_cmd_vel_threshold_;
    const bool actually_moving = actual_speed > stall_vel_raw_threshold_;

    if (!commanding_motion || actually_moving) {
      stall_since_.reset();
      return;
    }

    if (!stall_since_) {
      stall_since_ = now();
    } else if ((now() - *stall_since_).seconds() > stall_timeout_sec_) {
      handle_stall_detected();
      stall_since_.reset();
    }
  }

  void handle_stall_detected()
  {
    RCLCPP_ERROR(get_logger(),
      "STALL DETECTED near '%s': commanding motion but /vel_raw shows no movement "
      "for %.1fs; cancelling goal and stopping", current_waypoint_.name.c_str(), stall_timeout_sec_);
    if (active_goal_handle_) {
      action_client_->async_cancel_goal(active_goal_handle_);
      active_goal_handle_.reset();
    }
    send_stall_alert();
    transition(State::SLEEPING);
  }

  // Build and publish a JSON mail request for the Python mail_node. Single
  // choke point for every email this node sends (waypoint, stall, loop
  // boundary, cat recognized).
  void send_mail(const std::string & subject, const std::vector<std::string> & paths)
  {
    std::ostringstream oss;
    oss << "{\"subject\":\"" << subject << "\",";
    oss << "\"to\":\"" << mail_to_ << "\",";
    oss << "\"paths\":[";
    for (std::size_t i = 0; i < paths.size(); ++i) {
      if (i > 0) {
        oss << ",";
      }
      oss << "\"" << paths[i] << "\"";
    }
    oss << "]}";

    std_msgs::msg::String msg;
    msg.data = oss.str();
    mail_pub_->publish(msg);
    RCLCPP_INFO(get_logger(), "Published mail request \"%s\" (%zu attachments)",
                subject.c_str(), paths.size());
  }

  void send_stall_alert()
  {
    send_mail(mail_subject_ + " - STALL near " + current_waypoint_.name, {});
  }

  void send_waypoint_mail(const std::string & waypoint_name, const std::string & image_path)
  {
    send_mail(mail_subject_ + " - " + waypoint_name, {image_path});
  }

  void transition(State s)
  {
    state_ = s;
    publish_state();
  }

  void publish_state()
  {
    std_msgs::msg::String s;
    switch (state_) {
      case State::SLEEPING: s.data = "SLEEPING"; break;
      case State::WAKING: s.data = "WAKING"; break;
      case State::PATROLLING: s.data = "PATROLLING"; break;
      case State::CAPTURE: s.data = "CAPTURE"; break;
      case State::RETURNING: s.data = "RETURNING"; break;
      case State::INVESTIGATING: s.data = "INVESTIGATING"; break;
    }
    state_pub_->publish(s);
  }

private:
  // Parameters / config
  bool start_on_boot_{true};
  bool loop_patrol_{true};
  double patrol_period_sec_{1800.0};
  double patrol_pause_sec_{10.0};
  double settle_after_arrival_sec_{1.0};
  std::string image_topic_;
  std::string mail_request_topic_;
  std::string state_topic_;
  std::string image_save_dir_;
  std::string mail_subject_;
  std::string mail_to_;
  std::string map_frame_;
  std::string cmd_vel_topic_;
  std::string vel_raw_topic_;
  double stall_timeout_sec_{6.0};
  double stall_cmd_vel_threshold_{0.03};
  double stall_vel_raw_threshold_{0.02};
  std::string detections_topic_;
  bool cat_recognized_mail_enabled_{true};
  double cat_recognized_min_conf_{0.6};
  double cat_recognized_cooldown_sec_{30.0};
  bool loop_boundary_mail_enabled_{true};

  // State
  State state_{State::SLEEPING};
  bool wake_requested_{false};
  Pose2D home_;
  Pose2D current_waypoint_;
  std::vector<std::string> cycle_image_paths_;
  std::optional<rclcpp::Time> stall_since_;
  GoalHandle::SharedPtr active_goal_handle_;
  // label ("white"/"brown") -> time of last cat-recognized email (cooldown).
  std::unordered_map<std::string, rclcpp::Time> last_cat_mail_;

  // Concurrency
  rclcpp::CallbackGroup::SharedPtr fsm_group_;
  rclcpp::CallbackGroup::SharedPtr io_group_;

  // ROS entities
  rclcpp::TimerBase::SharedPtr fsm_timer_;
  rclcpp::TimerBase::SharedPtr wake_timer_;
  rclcpp::TimerBase::SharedPtr pause_timer_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
  rclcpp::Subscription<vision_msgs::msg::Detection2DArray>::SharedPtr detections_sub_;
  rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr cmd_vel_sub_;
  rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr vel_raw_sub_;
  geometry_msgs::msg::Twist::SharedPtr last_cmd_vel_;
  geometry_msgs::msg::Twist::SharedPtr last_vel_raw_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr mail_pub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr state_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr waypoints_marker_pub_;
  sensor_msgs::msg::Image::SharedPtr last_image_;
  rclcpp_action::Client<NavigateToPose>::SharedPtr action_client_;

  // Strategy
  std::unique_ptr<PatrolStrategy> strategy_;
};

}  // namespace patrol_manager

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);

  auto node = std::make_shared<patrol_manager::PatrolManagerNode>();
  rclcpp::executors::MultiThreadedExecutor exec(rclcpp::ExecutorOptions(), 2);
  exec.add_node(node);
  exec.spin();

  rclcpp::shutdown();
  return 0;
}
