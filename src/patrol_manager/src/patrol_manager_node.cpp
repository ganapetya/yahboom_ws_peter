#include <chrono>
#include <cmath>
#include <filesystem>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "geometry_msgs/msg/pose_stamped.hpp"
#include "nav2_msgs/action/navigate_to_pose.hpp"
#include "rclcpp/rclcpp.hpp"
#include "rclcpp_action/rclcpp_action.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "std_msgs/msg/string.hpp"

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
    settle_after_arrival_sec_ = declare_parameter<double>("settle_after_arrival_sec", 1.0);

    image_topic_ = declare_parameter<std::string>("image_topic", "/camera/color/image_raw");
    mail_request_topic_ = declare_parameter<std::string>("mail_request_topic", "/cat_patrol/mail_request");
    state_topic_ = declare_parameter<std::string>("state_topic", "/patrol_manager/state");
    image_save_dir_ = declare_parameter<std::string>("image_save_dir", "/tmp/cat_patrol_images");
    mail_subject_ = declare_parameter<std::string>("mail_subject", "Cat patrol photos");
    mail_to_ = declare_parameter<std::string>("smtp_to_address", "user@example.com");
    map_frame_ = declare_parameter<std::string>("map_frame", "map");

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

  void tick()
  {
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
        send_next_waypoint_goal();
        break;

      case State::PATROLLING:
        // Waiting for action result callback to transition.
        break;

      case State::CAPTURE:
        if (save_current_image()) {
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
    auto goal = NavigateToPose::Goal();
    goal.pose = to_pose(wp, map_frame_, now());

    transition(State::PATROLLING);
    RCLCPP_INFO(get_logger(), "Navigating to waypoint '%s' x=%.2f y=%.2f yaw=%.2f",
                wp.name.c_str(), wp.x, wp.y, wp.yaw);

    auto opts = rclcpp_action::Client<NavigateToPose>::SendGoalOptions();
    opts.feedback_callback =
      [this](GoalHandle::SharedPtr,
             const std::shared_ptr<const NavigateToPose::Feedback> fb) {
        RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 1000,
          "distance remaining: %.2f m", fb->distance_remaining);
      };

    opts.result_callback =
      [this](const GoalHandle::WrappedResult & result) {
        if (result.code == rclcpp_action::ResultCode::SUCCEEDED) {
          this->transition(State::CAPTURE);
        } else {
          RCLCPP_ERROR(this->get_logger(), "Waypoint goal failed (code=%d)", static_cast<int>(result.code));
          this->transition(State::SLEEPING);
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
    opts.result_callback =
      [this](const GoalHandle::WrappedResult & result) {
        if (result.code == rclcpp_action::ResultCode::SUCCEEDED) {
          send_mail_request();
          transition(State::SLEEPING);
          if (!loop_patrol_) {
            RCLCPP_INFO(get_logger(), "loop_patrol=false, cycle complete");
          }
        } else {
          RCLCPP_ERROR(get_logger(), "Home goal failed (code=%d)", static_cast<int>(result.code));
          transition(State::SLEEPING);
        }
      };

    action_client_->async_send_goal(goal, opts);
  }

  bool save_current_image()
  {
    if (!last_image_ || last_image_->data.empty()) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "No image available yet");
      return false;
    }

    try {
      // Manual sensor_msgs::Image -> cv::Mat conversion (same approach as
      // patrol_node.cpp). Avoids a cv_bridge dependency/ABI risk on this build.
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
      std::ostringstream fn;
      fn << "snap_" << now().nanoseconds() << ".jpg";
      const auto fp = (std::filesystem::path(image_save_dir_) / fn.str()).string();
      if (cv::imwrite(fp, bgr_image)) {
        cycle_image_paths_.push_back(fp);
        RCLCPP_INFO(get_logger(), "Captured: %s", fp.c_str());
        return true;
      }
    } catch (const std::exception & e) {
      RCLCPP_WARN(get_logger(), "save_current_image exception: %s", e.what());
    }
    return false;
  }

  void send_mail_request()
  {
    std::ostringstream oss;
    oss << "{\"subject\":\"" << mail_subject_ << "\",";
    oss << "\"to\":\"" << mail_to_ << "\",";
    oss << "\"paths\":[";
    for (size_t i = 0; i < cycle_image_paths_.size(); ++i) {
      if (i > 0) {oss << ",";}
      oss << "\"" << cycle_image_paths_[i] << "\"";
    }
    oss << "]}";

    std_msgs::msg::String msg;
    msg.data = oss.str();
    mail_pub_->publish(msg);
    RCLCPP_INFO(get_logger(), "Published mail request with %zu images", cycle_image_paths_.size());
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
  double settle_after_arrival_sec_{1.0};
  std::string image_topic_;
  std::string mail_request_topic_;
  std::string state_topic_;
  std::string image_save_dir_;
  std::string mail_subject_;
  std::string mail_to_;
  std::string map_frame_;

  // State
  State state_{State::SLEEPING};
  bool wake_requested_{false};
  Pose2D home_;
  std::vector<std::string> cycle_image_paths_;

  // Concurrency
  rclcpp::CallbackGroup::SharedPtr fsm_group_;
  rclcpp::CallbackGroup::SharedPtr io_group_;

  // ROS entities
  rclcpp::TimerBase::SharedPtr fsm_timer_;
  rclcpp::TimerBase::SharedPtr wake_timer_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr mail_pub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr state_pub_;
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
