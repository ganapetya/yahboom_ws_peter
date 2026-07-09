#ifndef PATROL_MANAGER__PATROL_STRATEGY_HPP_
#define PATROL_MANAGER__PATROL_STRATEGY_HPP_

#include <string>
#include <vector>

namespace patrol_manager
{

struct Pose2D
{
  double x{0.0};
  double y{0.0};
  double yaw{0.0};
  std::string name;
};

class PatrolStrategy
{
public:
  virtual ~PatrolStrategy() = default;
  virtual std::string name() const = 0;
  virtual void reset() = 0;
  virtual bool has_next() const = 0;
  virtual Pose2D next() = 0;
};

}  // namespace patrol_manager

#endif  // PATROL_MANAGER__PATROL_STRATEGY_HPP_
