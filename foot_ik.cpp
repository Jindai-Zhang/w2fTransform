#include "foot_ik.h"

#include <algorithm>
#include <cmath>

namespace
{
bool Finite(const mjtNum* values, int count)
{
  for (int i = 0; i < count; ++i)
    if (!std::isfinite(values[i])) return false;
  return true;
}

bool IsAncestor(const mjModel* model, int ancestor, int body)
{
  for (; body > 0; body = model->body_parentid[body])
    if (body == ancestor) return true;
  return false;
}

// 世界坐标角速度对应的旋转误差：R_target * R_current^T。
void RotationError(const double* target, const mjtNum* current, mjtNum* result)
{
  mjtNum inverse[4], difference[4];
  mju_negQuat(inverse, current);
  mju_mulQuat(difference, target, inverse);
  if (difference[0] < 0.0)
    for (mjtNum& value : difference) value = -value;
  mju_quat2Vel(result, difference, 1.0);
}

double Distance3(const mjtNum* actual, const double* desired)
{
  double sum = 0.0;
  for (int i = 0; i < 3; ++i) sum += (actual[i]-desired[i])*(actual[i]-desired[i]);
  return std::sqrt(sum);
}
} // namespace

FootIK::~FootIK()
{
  Reset();
}

void FootIK::Reset()
{
  if (reference_) mj_deleteData(reference_);
  reference_ = nullptr;
  model_ = nullptr;
  elapsed_ = 0.0;
  velocity_.fill(0.0);
  diagnostics_ = {};
}

bool FootIK::Initialize(const mjModel* model, const mjData* actual, int torso,
                        int imu_site, const std::array<int, 8>& joint_ids)
{
  Reset();
  if (!model || !actual || torso <= 0 || torso >= model->nbody ||
      imu_site < 0 || imu_site >= model->nsite ||
      !IsAncestor(model, torso, model->site_bodyid[imu_site]) ||
      !Finite(actual->qpos, model->nq)) return false;

  int root = -1;
  for (int joint = 0; joint < model->njnt; ++joint)
    if (model->jnt_type[joint] == mjJNT_FREE &&
        IsAncestor(model, model->jnt_bodyid[joint], torso))
    {
      if (root != -1) return false;
      root = joint;
    }
  if (root < 0) return false;
  for (int i = 0; i < 6; ++i) dof_addresses_[i] = model->jnt_dofadr[root]+i;

  for (int i = 0; i < 8; ++i)
  {
    const int joint = joint_ids[i];
    if (joint < 0 || joint >= model->njnt || model->jnt_type[joint] != mjJNT_HINGE ||
        !IsAncestor(model, torso, model->jnt_bodyid[joint])) return false;
    for (int previous = 0; previous < i; ++previous)
      if (joint_ids[previous] == joint) return false;
    qpos_addresses_[i] = model->jnt_qposadr[joint];
    dof_addresses_[i+6] = model->jnt_dofadr[joint];
    const double position = actual->qpos[qpos_addresses_[i]];
    if (model->jnt_limited[joint] &&
        (position < model->jnt_range[2*joint]-1e-7 ||
         position > model->jnt_range[2*joint+1]+1e-7)) return false;
  }

  reference_ = mj_makeData(model);
  if (!reference_) return false;
  model_ = model;
  torso_ = torso;
  imu_site_ = imu_site;
  joints_ = joint_ids;
  feet_ = {model->jnt_bodyid[joint_ids[3]], model->jnt_bodyid[joint_ids[7]]};
  if (feet_[0] == feet_[1]) { Reset(); return false; }
  mj_copyData(reference_, model_, actual);
  jacobian_position_.assign(3*model->nv, 0.0);
  jacobian_rotation_.assign(3*model->nv, 0.0);
  full_velocity_.assign(model->nv, 0.0);
  saved_qpos_.assign(model->nq, 0.0);
  ForwardKinematics();

  for (int side = 0; side < 2; ++side)
  {
    std::copy_n(reference_->xpos+3*feet_[side], 3, foot_positions_[side].begin());
    std::copy_n(reference_->xquat+4*feet_[side], 4, foot_rotations_[side].begin());
  }
  std::copy_n(reference_->xquat+4*torso_, 4, torso_rotation_.begin());
  ReadTargets(initial_joint_positions_);
  if (!Finite(reference_->subtree_com+3*torso_, 3)) { Reset(); return false; }
  return true;
}

void FootIK::ForwardKinematics()
{
  // 不调用 mj_forward/mj_step，不触发控制回调；所有被动形态坐标保持初始化值。
  mj_kinematics(model_, reference_);
  mj_comPos(model_, reference_);
}

void FootIK::ReadTargets(std::array<double, 8>& result) const
{
  for (int i = 0; i < 8; ++i) result[i] = reference_->qpos[qpos_addresses_[i]];
}

void FootIK::AddTaskRow(const mjtNum* jacobian_row, double desired_velocity,
                        double weight)
{
  for (int row = 0; row < kVariables; ++row)
  {
    const double entry = jacobian_row[dof_addresses_[row]];
    gradient_[row] -= weight*entry*desired_velocity;
    for (int col = 0; col < kVariables; ++col)
      hessian_[row*kVariables+col] +=
          weight*entry*jacobian_row[dof_addresses_[col]];
  }
}

void FootIK::AddBodyPoseTask(int body, const std::array<double, 3>& position,
                            const std::array<double, 4>& rotation)
{
  mj_jacBody(model_, reference_, jacobian_position_.data(), jacobian_rotation_.data(), body);
  mjtNum rotation_error[3];
  RotationError(rotation.data(), reference_->xquat+4*body, rotation_error);
  for (int axis = 0; axis < 3; ++axis)
  {
    const double linear = std::clamp(15.0*(position[axis]-reference_->xpos[3*body+axis]), -0.04, 0.04);
    const double angular = std::clamp(15.0*rotation_error[axis], -0.15, 0.15);
    AddTaskRow(jacobian_position_.data()+axis*model_->nv, linear, 200000.0);
    AddTaskRow(jacobian_rotation_.data()+axis*model_->nv, angular, 20000.0);
  }
}

void FootIK::UpdateDiagnostics(const std::array<double, 3>& target_com,
                               double target_chest_z)
{
  diagnostics_.foot_position_error = 0.0;
  diagnostics_.foot_rotation_error = 0.0;
  for (int side = 0; side < 2; ++side)
  {
    diagnostics_.foot_position_error = std::max(diagnostics_.foot_position_error,
        Distance3(reference_->xpos+3*feet_[side], foot_positions_[side].data()));
    mjtNum error[3];
    RotationError(foot_rotations_[side].data(), reference_->xquat+4*feet_[side], error);
    diagnostics_.foot_rotation_error = std::max(diagnostics_.foot_rotation_error,
        static_cast<double>(mju_norm3(error)));
  }
  mjtNum torso_error[3];
  RotationError(torso_rotation_.data(), reference_->xquat+4*torso_, torso_error);
  diagnostics_.torso_rotation_error = mju_norm3(torso_error);
  const mjtNum* com = reference_->subtree_com+3*torso_;
  diagnostics_.com_xy_error = std::hypot(com[0]-target_com[0], com[1]-target_com[1]);
  diagnostics_.chest_height_error = reference_->site_xpos[3*imu_site_+2]-target_chest_z;
}

bool FootIK::Step(const mjData* actual, double dt,
                  const std::array<double, 3>& target_com, double target_chest_z,
                  std::array<double, 8>& joint_targets)
{
  diagnostics_.updated = false;
  if (!reference_) return false;
  ReadTargets(joint_targets);
  if (!actual || !std::isfinite(dt) || dt <= 0.0 || dt > 0.1 ||
      !std::isfinite(target_chest_z) || !Finite(target_com.data(), 3) ||
      !Finite(actual->subtree_com+3*torso_, 3) ||
      !Finite(actual->site_xpos+3*imu_site_, 3)) return false;
  elapsed_ += dt;
  if (elapsed_+1e-12 < kPeriod) return true;
  // 不追赶暂停/卡顿后的长时间步；每次最大只推进 20 ms 的有界参考。
  elapsed_ = std::max(0.0, elapsed_-kPeriod*std::floor((elapsed_+1e-12)/kPeriod));
  hessian_.fill(0.0);
  gradient_.fill(0.0);

  for (int side = 0; side < 2; ++side)
    AddBodyPoseTask(feet_[side], foot_positions_[side], foot_rotations_[side]);

  mj_jacBody(model_, reference_, nullptr, jacobian_rotation_.data(), torso_);
  mjtNum rotation_error[3];
  RotationError(torso_rotation_.data(), reference_->xquat+4*torso_, rotation_error);
  for (int axis = 0; axis < 3; ++axis)
    AddTaskRow(jacobian_rotation_.data()+axis*model_->nv,
               std::clamp(6.0*rotation_error[axis], -0.12, 0.12), 2000.0);

  mj_jacSubtreeCom(model_, reference_, jacobian_position_.data(), torso_);
  const mjtNum* com = reference_->subtree_com+3*torso_;
  const mjtNum* actual_com = actual->subtree_com+3*torso_;
  for (int axis = 0; axis < 2; ++axis)
  {
    const double compensation = std::clamp(0.20*(com[axis]-actual_com[axis]), -0.005, 0.005);
    const double desired = std::clamp(3.0*(target_com[axis]+compensation-com[axis]), -0.025, 0.025);
    AddTaskRow(jacobian_position_.data()+axis*model_->nv, desired, 200.0);
  }
  mj_jacSite(model_, reference_, jacobian_position_.data(), nullptr, imu_site_);
  const double chest = reference_->site_xpos[3*imu_site_+2];
  const double compensation = std::clamp(0.20*(chest-actual->site_xpos[3*imu_site_+2]), -0.003, 0.003);
  AddTaskRow(jacobian_position_.data()+2*model_->nv,
      std::clamp(4.0*(target_chest_z+compensation-chest), -0.02, 0.02), 1000.0);

  for (int i = 0; i < kVariables; ++i)
  {
    // 严格正定正规化，同时弱惩罚关节偏离初始化构型。
    hessian_[i*kVariables+i] += 0.05;
    const double limit = i < 3 ? 0.05 : (i < 6 ? 0.15 : 0.35);
    lower_[i] = -limit;
    upper_[i] = limit;
    if (i >= 6)
    {
      const int index = i-6;
      const int joint = joints_[index];
      const double q = reference_->qpos[qpos_addresses_[index]];
      gradient_[i] -= 0.005*(initial_joint_positions_[index]-q);
      if (model_->jnt_limited[joint])
      {
        lower_[i] = std::max(lower_[i], (model_->jnt_range[2*joint]-q)/kPeriod);
        upper_[i] = std::min(upper_[i], (model_->jnt_range[2*joint+1]-q)/kPeriod);
      }
    }
    if (lower_[i] > upper_[i]) return false;
    velocity_[i] = std::clamp(velocity_[i], lower_[i], upper_[i]);
  }
  if (!Finite(hessian_.data(), kVariables*kVariables) ||
      !Finite(gradient_.data(), kVariables)) return false;
  diagnostics_.solver_free_variables = mju_boxQP(velocity_.data(), factor_.data(),
      free_indices_.data(), hessian_.data(), gradient_.data(), kVariables,
      lower_.data(), upper_.data());
  if (diagnostics_.solver_free_variables < 0 || !Finite(velocity_.data(), kVariables))
  {
    velocity_.fill(0.0);
    return false;
  }

  std::copy_n(reference_->qpos, model_->nq, saved_qpos_.begin());
  std::fill(full_velocity_.begin(), full_velocity_.end(), 0.0);
  for (int i = 0; i < kVariables; ++i) full_velocity_[dof_addresses_[i]] = velocity_[i];
  bool accepted = false;
  double scale = 1.0;
  for (int attempt = 0; attempt < 7; ++attempt, scale *= 0.5)
  {
    std::copy(saved_qpos_.begin(), saved_qpos_.end(), reference_->qpos);
    mj_integratePos(model_, reference_->qpos, full_velocity_.data(), kPeriod*scale);
    ForwardKinematics();
    UpdateDiagnostics(target_com, target_chest_z);
    // 高权重足端任务并非硬等式；这里拒绝参考的累积足端漂移。
    if (Finite(reference_->qpos, model_->nq) &&
        std::isfinite(diagnostics_.foot_position_error) &&
        std::isfinite(diagnostics_.foot_rotation_error) &&
        diagnostics_.foot_position_error <= 0.001 &&
        diagnostics_.foot_rotation_error <= 0.005 &&
        diagnostics_.torso_rotation_error <= 0.02)
    {
      accepted = true;
      break;
    }
  }
  if (!accepted)
  {
    std::copy(saved_qpos_.begin(), saved_qpos_.end(), reference_->qpos);
    ForwardKinematics();
    UpdateDiagnostics(target_com, target_chest_z);
    velocity_.fill(0.0);
    return false;
  }
  for (mjtNum& value : velocity_) value *= scale;
  diagnostics_.updated = true;
  ReadTargets(joint_targets);
  return true;
}
