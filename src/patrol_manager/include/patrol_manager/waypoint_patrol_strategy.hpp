#ifndef PATROL_MANAGER__WAYPOINT_PATROL_STRATEGY_HPP_
#define PATROL_MANAGER__WAYPOINT_PATROL_STRATEGY_HPP_

#include <cstddef>
#include <vector>

#include "patrol_manager/patrol_strategy.hpp"

namespace patrol_manager
{

class WaypointPatrolStrategy : public PatrolStrategy
{
public:
  explicit WaypointPatrolStrategy(std::vector<Pose2D> waypoints);

  std::string name() const override;
  void reset() override;
  bool has_next() const override;
  Pose2D next() override;

private:
  std::vector<Pose2D> waypoints_;
  std::size_t index_{0};
};

}  // namespace patrol_manager

#endif  // PATROL_MANAGER__WAYPOINT_PATROL_STRATEGY_HPP_
