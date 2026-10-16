#pragma once

#include <mujoco/mujoco.h>

// Quasi-static actuator-side force in N for one NUT slide (side 0=right, 1=left).
// Holds the floating base, leg joints and ATS fixed while differentiating the
// six passive module joints along the equality-constrained shape coordinate.
// Uses current bias/passive/applied forces and measured contact reactions only;
// joint-limit reactions are deliberately NOT cancelled by this feedforward.
//
// This function does not change mjModel/mjData or advance the simulation. At an
// mjcb_control call, contact reactions belong to the previous constraint solve.
// A consistency check rejects substantial force/Jacobian topology mismatches.
// The caller must filter/slew/limit accepted estimates and retain a safe prior
// estimate on false; this is not a replacement for shape-position feedback.
// residual is ||Je*v||_2 / max(1, ||Je[:,nut]||_2), with v[nut]=1 m/s.
// force is set to zero on failure. No M*qacc term is included.
bool ComputeShapeHoldingForce(const mjModel* model, const mjData* data,
                              int side, double& force, double& residual);
