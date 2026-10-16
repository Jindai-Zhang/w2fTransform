#include "shape_feedforward.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <vector>

namespace {
constexpr int kPassive = 6;
constexpr int kActiveColumns = kPassive + 1;
using Row = std::array<mjtNum, kActiveColumns>;

bool ContactRow(int type) {
  return type == mjCNSTR_CONTACT_FRICTIONLESS ||
         type == mjCNSTR_CONTACT_PYRAMIDAL ||
         type == mjCNSTR_CONTACT_ELLIPTIC;
}

// Extract only NUT and its passive module columns; all other virtual velocities
// are zero. Sparse and dense compiled models use the same public interface.
Row ExtractRow(const mjModel* model, const mjData* data, int row,
               const std::array<int, kActiveColumns>& dofs, bool sparse) {
  Row result{};
  if (sparse) {
    const int begin = data->efc_J_rowadr[row];
    const int end = begin + data->efc_J_rownnz[row];
    for (int k = begin; k < end; ++k)
      for (int j = 0; j < kActiveColumns; ++j)
        if (data->efc_J_colind[k] == dofs[j]) result[j] = data->efc_J[k];
  } else {
    for (int j = 0; j < kActiveColumns; ++j)
      result[j] = data->efc_J[row*model->nv+dofs[j]];
  }
  return result;
}

// In a control callback the new geometry/J is available before the next force
// solve. Check J'lambda against the previous generalized constraint force, whose
// DOF indexing is stable. This catches reassigned efc rows after contact changes.
bool ConstraintForcesConsistent(const mjModel* model, const mjData* data, bool sparse) {
  std::vector<mjtNum> reconstructed(model->nv, 0);
  for (int row = 0; row < data->nefc; ++row) {
    const mjtNum lambda = data->efc_force[row];
    if (!std::isfinite(lambda)) return false;
    if (sparse) {
      const int begin = data->efc_J_rowadr[row];
      const int end = begin+data->efc_J_rownnz[row];
      for (int k = begin; k < end; ++k)
        reconstructed[data->efc_J_colind[k]] += data->efc_J[k]*lambda;
    } else {
      for (int j = 0; j < model->nv; ++j)
        reconstructed[j] += data->efc_J[row*model->nv+j]*lambda;
    }
  }
  double difference2 = 0, reference2 = 0;
  for (int j = 0; j < model->nv; ++j) {
    if (!std::isfinite(reconstructed[j]) || !std::isfinite(data->qfrc_constraint[j]))
      return false;
    const double difference = reconstructed[j]-data->qfrc_constraint[j];
    difference2 += difference*difference;
    reference2 += data->qfrc_constraint[j]*data->qfrc_constraint[j];
  }
  // A small one-step geometry drift is acceptable; a changed force layout is not.
  return std::sqrt(difference2) <= 0.02*(1.0+std::sqrt(reference2));
}
}  // namespace

bool ComputeShapeHoldingForce(const mjModel* model, const mjData* data,
                              int side, double& force, double& residual) {
  force = 0;
  residual = std::numeric_limits<double>::infinity();
  if (!model || !data || side < 0 || side > 1 || data->nefc < kPassive || model->nv <= 0)
    return false;

  static constexpr const char* names[2][kActiveColumns] = {
      {"rNUT_slide", "rPTS_roll", "rALK_pitch", "rATO_pitch", "rGRD_roll", "rPLK_pitch", "rPTO_pitch"},
      {"lNUT_slide", "lPTS_roll", "lALK_pitch", "lATO_pitch", "lGRD_roll", "lPLK_pitch", "lPTO_pitch"}};
  std::array<int, kActiveColumns> dofs{};
  for (int j = 0; j < kActiveColumns; ++j) {
    const int joint = mj_name2id(model, mjOBJ_JOINT, names[side][j]);
    if (joint < 0 || model->jnt_type[joint] != (j == 0 ? mjJNT_SLIDE : mjJNT_HINGE))
      return false;
    dofs[j] = model->jnt_dofadr[joint];
  }
  const bool sparse = mj_isSparse(model) != 0;
  std::vector<Row> equalities;
  equalities.reserve(data->nefc);
  std::array<mjtNum, kPassive*kPassive> normal{};
  std::array<mjtNum, kPassive> rhs{};
  double nut_column2 = 0;
  for (int row = 0; row < data->nefc; ++row) {
    if (data->efc_type[row] != mjCNSTR_EQUALITY) continue;
    const auto values = ExtractRow(model, data, row, dofs, sparse);
    for (const double value : values) if (!std::isfinite(value)) return false;
    equalities.push_back(values);
    nut_column2 += values[0]*values[0];
    for (int i = 0; i < kPassive; ++i) {
      rhs[i] -= values[i+1]*values[0];
      for (int j = 0; j < kPassive; ++j)
        normal[kPassive*i+j] += values[i+1]*values[j+1];
    }
  }
  if (equalities.size() < kPassive || nut_column2 < 1e-12) return false;

  // Column scaling makes the rank/damping check independent of the markedly
  // different metre-valued connect rows and radian-valued groove rows.
  std::array<mjtNum, kPassive> scale{};
  for (int i = 0; i < kPassive; ++i) {
    if (!(normal[kPassive*i+i] > 1e-14)) return false;
    scale[i] = std::sqrt(normal[kPassive*i+i]);
  }
  for (int i = 0; i < kPassive; ++i) {
    rhs[i] /= scale[i];
    for (int j = 0; j < kPassive; ++j) normal[kPassive*i+j] /= scale[i]*scale[j];
  }
  // Test rank BEFORE regularization: damping must not hide a mechanism singularity.
  auto rank_test = normal;
  if (mju_cholFactor(rank_test.data(), kPassive, 1e-9) != kPassive) return false;
  for (int i = 0; i < kPassive; ++i) normal[kPassive*i+i] += 1e-10;
  if (mju_cholFactor(normal.data(), kPassive, 1e-14) != kPassive) return false;
  std::array<mjtNum, kPassive> scaled_velocity{};
  mju_cholSolve(scaled_velocity.data(), normal.data(), rhs.data(), kPassive);
  Row velocity{};
  velocity[0] = 1;
  for (int i = 0; i < kPassive; ++i) {
    velocity[i+1] = scaled_velocity[i]/scale[i];
    if (!std::isfinite(velocity[i+1]) || std::abs(velocity[i+1]) > 1e4) return false;
  }
  double error2 = 0;
  for (const auto& row : equalities) {
    double error = 0;
    for (int j = 0; j < kActiveColumns; ++j) error += row[j]*velocity[j];
    error2 += error*error;
  }
  residual = std::sqrt(error2)/std::max(1.0, std::sqrt(nut_column2));
  if (!std::isfinite(residual) || residual > 1e-4 ||
      !ConstraintForcesConsistent(model, data, sparse)) return false;

  double holding = 0;
  for (int j = 0; j < kActiveColumns; ++j) {
    const int dof = dofs[j];
    holding += velocity[j]*(data->qfrc_bias[dof]-data->qfrc_passive[dof]-data->qfrc_applied[dof]);
  }
  // J_contact' * lambda already includes all tangential, torsional and rolling
  // components, including the pyramid representation. Exclude equality and
  // joint-limit rows so neither numerical closure force nor stop force is fed back.
  for (int row = 0; row < data->nefc; ++row) {
    if (!ContactRow(data->efc_type[row])) continue;
    const auto jacobian = ExtractRow(model, data, row, dofs, sparse);
    double virtual_velocity = 0;
    for (int j = 0; j < kActiveColumns; ++j) virtual_velocity += jacobian[j]*velocity[j];
    holding -= virtual_velocity*data->efc_force[row];
  }
  // Applied Cartesian forces are uncommon in normal operation but matter during
  // disturbance tests. Compute their virtual work without modifying the data.
  std::vector<mjtNum> jacp, jacr;
  for (int body = 1; body < model->nbody; ++body) {
    const auto* wrench = data->xfrc_applied+6*body;
    bool nonzero = false;
    for (int j = 0; j < 6; ++j) nonzero = nonzero || wrench[j] != 0;
    if (!nonzero) continue;
    if (jacp.empty()) { jacp.resize(3*model->nv); jacr.resize(3*model->nv); }
    mj_jac(model, data, jacp.data(), jacr.data(), data->xipos+3*body, body);
    for (int axis = 0; axis < 3; ++axis)
      for (int j = 0; j < kActiveColumns; ++j)
        holding -= velocity[j]*(jacp[axis*model->nv+dofs[j]]*wrench[axis] +
                                jacr[axis*model->nv+dofs[j]]*wrench[axis+3]);
  }
  if (!std::isfinite(holding)) return false;
  force = holding;
  return true;
}
