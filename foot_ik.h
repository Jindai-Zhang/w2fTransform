#pragma once

#include <mujoco/mujoco.h>

#include <array>
#include <vector>

// 双足承重后的参考构型生成器。输入/输出采用 rad、m、s。
// 这是带速度和关节位置界的运动学任务 QP，不约束力矩、接触力或摩擦。
// ATS 位姿通过高权重任务和漂移检查保持；外部状态机负责判断实际接触、
// 形态保持和负载能力，并负责用力矩/位置闭环跟踪输出参考。
class FootIK
{
public:
  struct Diagnostics
  {
    bool updated = false;       // 本次调用是否真正执行了 20 ms 参考更新
    int solver_free_variables = 0;
    double foot_position_error = 0.0;  // 参考 FK 的最大足端位置误差，m
    double foot_rotation_error = 0.0;  // 参考 FK 的最大足端角度误差，rad
    double torso_rotation_error = 0.0;
    double com_xy_error = 0.0;         // 参考相对本次输入目标的误差，m
    double chest_height_error = 0.0;   // 参考相对本次输入目标的误差，m
  };

  FootIK() = default;
  ~FootIK();
  FootIK(const FootIK&) = delete;
  FootIK& operator=(const FootIK&) = delete;
  FootIK(FootIK&&) = delete;
  FootIK& operator=(FootIK&&) = delete;

  // 必须在形态展开并建立稳定足支撑后初始化。顺序固定为
  // [Rhip, Rfem, Rtib, RATS, Lhip, Lfem, Ltib, LATS] 的 MuJoCo joint ID。
  // 模型应比本对象活得更久；mjData 独立持有，不修改 actual。
  bool Initialize(const mjModel* model, const mjData* actual, int torso,
                  int imu_site, const std::array<int, 8>& joint_ids);

  // target_com 的 x/y 用于全身 CoM 任务；z 不参与任务，垂向任务使用胸高。
  // 每 0.02 s 更新一次，其余调用返回上次参考。true 只表示求解/保持有效，
  // 不表示已到目标或实际机器人满足接触/动力学条件；失败时保留上次参考。
  bool Step(const mjData* actual, double dt,
            const std::array<double, 3>& target_com, double target_chest_z,
            std::array<double, 8>& joint_targets);

  bool initialized() const { return reference_ != nullptr; }
  const Diagnostics& diagnostics() const { return diagnostics_; }

private:
  static constexpr int kVariables = 14;
  static constexpr double kPeriod = 0.02;

  void Reset();
  void ForwardKinematics();
  void ReadTargets(std::array<double, 8>& result) const;
  void AddTaskRow(const mjtNum* jacobian_row, double desired_velocity,
                  double weight);
  void AddBodyPoseTask(int body, const std::array<double, 3>& position,
                      const std::array<double, 4>& rotation);
  void UpdateDiagnostics(const std::array<double, 3>& target_com,
                         double target_chest_z);

  const mjModel* model_ = nullptr;
  mjData* reference_ = nullptr;
  int torso_ = -1;
  int imu_site_ = -1;
  std::array<int, 8> joints_{};
  std::array<int, 8> qpos_addresses_{};
  std::array<int, kVariables> dof_addresses_{};
  std::array<int, 2> feet_{};
  std::array<std::array<double, 3>, 2> foot_positions_{};
  std::array<std::array<double, 4>, 2> foot_rotations_{};
  std::array<double, 4> torso_rotation_{};
  std::array<double, 8> initial_joint_positions_{};

  std::array<mjtNum, kVariables * kVariables> hessian_{};
  std::array<mjtNum, kVariables> gradient_{};
  std::array<mjtNum, kVariables> velocity_{};
  std::array<mjtNum, kVariables> lower_{};
  std::array<mjtNum, kVariables> upper_{};
  std::array<mjtNum, kVariables * (kVariables + 7)> factor_{};
  std::array<int, kVariables> free_indices_{};
  std::vector<mjtNum> jacobian_position_;
  std::vector<mjtNum> jacobian_rotation_;
  std::vector<mjtNum> full_velocity_;
  std::vector<mjtNum> saved_qpos_;
  double elapsed_ = 0.0;
  Diagnostics diagnostics_{};
};
