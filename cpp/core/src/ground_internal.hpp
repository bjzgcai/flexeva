#pragma once
#include "runtime_internal.hpp"

namespace flexeva::core::runtime::detail {
// One session belongs to one transaction. Only the selected parent is a cache
// source. Its request cache survives untouched rounds; operation dedup is local.
struct GroundSession;
void validate_ground_registrations(const Pipeline&);
Evidence ground_policy_identity(const std::vector<GroundRegistration>&);
std::shared_ptr<GroundSession> make_ground_session(
    ras::Transaction&, ras::DirectoryId current, ras::ReadView parent,
    ras::DirectoryId parent_facts, const std::vector<GroundRegistration>&, Id window);
} // namespace flexeva::core::runtime::detail
