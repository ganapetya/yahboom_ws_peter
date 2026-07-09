#include "patrol_manager/waypoint_patrol_strategy.hpp"

#include <stdexcept>

namespace patrol_manager
{

WaypointPatrolStrategy::WaypointPatrolStrategy(std::vector<Pose2D> waypoints)
: waypoints_(std::move(waypoints))
{}

std::string WaypointPatrolStrategy::name() const
{
  return "waypoint_patrol";
}

void WaypointPatrolStrategy::reset()
{
  index_ = 0;
}

bool WaypointPatrolStrategy::has_next() const
{
  return index_ < waypoints_.size();
}

Pose2D WaypointPatrolStrategy::next()
{
  if (!has_next()) {
    throw std::runtime_error("WaypointPatrolStrategy::next called with no remaining waypoints");
  }
  return waypoints_[index_++];
}

}  // namespace patrol_manager
