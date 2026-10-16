#include "user_func.h"
#include "foot_ik.h"
#include "shape_feedforward.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {
constexpr double pi = 3.14159265358979323846, deg = pi / 180.0;
constexpr double stroke = 0.040878;
double Limit(double x, double a) { return std::clamp(x, -a, a); }
double Smooth(double x) { x = std::clamp(x, 0.0, 1.0); return x*x*x*(10+x*(-15+6*x)); }
double SmoothRate(double x) { x=std::clamp(x,0.0,1.0); return 30*x*x*(1-x)*(1-x); }
std::array<double,4> LqrGains(double total_mass) {
  // 轮腿倒立摆标称模型；扭矩是每侧ATS关节力矩，轮轴正方向为-Y。
  const double mass=total_mass-4.24,h=0.46,inertia=2.0,r=0.138,dt=0.002;
  const double d=total_mass*(inertia+mass*h*h)-mass*mass*h*h;
  const double ax=-mass*h*mass*9.81*h/d,ap=total_mass*mass*9.81*h/d;
  const double bx=(-(inertia+mass*h*h)*2/r-mass*h*2)/d;
  const double bp=(total_mass*2+mass*h*2/r)/d;
  double a[4][4]{{1,dt,0,0},{0,1,dt*ax,0},{0,0,1,dt},{0,0,dt*ap,1}};
  double b[4]{0,dt*bx,0,dt*bp},q[4]{100,10,400,30},p[4][4]{};
  std::array<double,4> k{};
  for(int i=0;i<4;++i) p[i][i]=q[i];
  for(int iter=0;iter<12000;++iter) {
    double pa[4][4]{},pb[4]{},next[4][4]{},den=0.1;
    for(int i=0;i<4;++i) for(int j=0;j<4;++j) {
      pb[i]+=p[i][j]*b[j];
      for(int l=0;l<4;++l) pa[i][j]+=p[i][l]*a[l][j];
    }
    for(int i=0;i<4;++i) den+=b[i]*pb[i];
    for(int j=0;j<4;++j) { k[j]=0; for(int i=0;i<4;++i) k[j]+=b[i]*pa[i][j]/den; }
    for(int i=0;i<4;++i) for(int j=0;j<4;++j) {
      next[i][j]=(i==j?q[i]:0)-den*k[i]*k[j];
      for(int l=0;l<4;++l) next[i][j]+=a[l][i]*pa[l][j];
    }
    double change=0;
    for(int i=0;i<4;++i) for(int j=0;j<4;++j) {
      const double v=0.5*(next[i][j]+next[j][i]);
      change=std::max(change,std::abs(v-p[i][j])); p[i][j]=v;
    }
    if(change<1e-7) break;
  }
  return k;
}
struct Point { double x,y; };
double Cross(Point a,Point b,Point c) { return (b.x-a.x)*(c.y-a.y)-(b.y-a.y)*(c.x-a.x); }
double HullMargin(std::vector<Point> pts, Point com) {
  if(pts.size()<3) return -1;
  std::sort(pts.begin(),pts.end(),[](Point a,Point b){return a.x==b.x?a.y<b.y:a.x<b.x;});
  std::vector<Point> h;
  for(auto p:pts) { while(h.size()>1 && Cross(h[h.size()-2],h.back(),p)<=1e-12) h.pop_back(); h.push_back(p); }
  const auto lower=h.size();
  for(int i=static_cast<int>(pts.size())-2;i>=0;--i) { while(h.size()>lower && Cross(h[h.size()-2],h.back(),pts[i])<=1e-12) h.pop_back(); h.push_back(pts[i]); }
  if(h.size()<4) return -1;
  double result=1;
  for(std::size_t i=0;i+1<h.size();++i) {
    const double len=std::hypot(h[i+1].x-h[i].x,h[i+1].y-h[i].y);
    if(len>1e-9) result=std::min(result,Cross(h[i],h[i+1],com)/len);
  }
  return result;
}

// 编译后的mesh顶点处于geom局部坐标，包含MuJoCo的网格重心坐标变换。
double MeshGap(const mjModel* m,const mjData* d,int geom) {
  const int mesh=m->geom_dataid[geom];
  const auto* r=d->geom_xmat+9*geom;
  const auto* p=d->geom_xpos+3*geom;
  double z=1e3;
  for(int k=0;k<m->mesh_vertnum[mesh];++k) {
    const auto* v=m->mesh_vert+3*(m->mesh_vertadr[mesh]+k);
    z=std::min(z,p[2]+r[6]*v[0]+r[7]*v[1]+r[8]*v[2]);
  }
  return z;
}
void Enter(UserData& u,TransitionStage stage) {
  auto& t=u.transition;
  t.stage=stage; t.stage_started=u.data->time; t.settled=0;
  if(stage==TransitionStage::Hold) {
    t.phase_goal=t.phase;
    t.nut_velocity=0;
    t.common_velocity_target=0;
    t.previous_common_velocity=0;
    t.differential_torque=0;
  }
  std::cout << "[transition " << u.data->time << "s] " << TransitionStageName(stage)
            << " phase=" << std::remainder(t.phase[0],2*pi)/deg << ","
            << std::remainder(t.phase[1],2*pi)/deg << " deg, nut="
            << u.joints[4]->GetActualPosition()*1000 << ","
            << u.joints[9]->GetActualPosition()*1000 << " mm\n";
}
void Observe(UserData& u,double dt,double pitch_rate) {
  auto& t=u.transition; const auto* d=u.data; const auto* m=u.model;
  const double yaw=u.imu->GetYaw(), ex=std::cos(yaw),ey=std::sin(yaw);
  for(int i=0;i<2;++i) {
    const auto* a=d->xmat+9*t.ats_body[i]; const auto* b=d->xmat+9*t.ato_body[i];
    const double phase=std::atan2(-a[6],ex*a[0]+ey*a[3]);
    if(!t.observed) t.phase[i]=phase;
    else t.phase[i]+=std::remainder(phase-t.raw_phase[i],2*pi);
    t.raw_phase[i]=phase;
    const int j=i*5;
    t.phase_rate[i]=pitch_rate-u.joints[j+1]->GetActualVelocity()-u.joints[j+2]->GetActualVelocity()-u.joints[j+3]->GetActualVelocity();
    t.alpha[i]=std::atan2(a[0]*b[2]+a[3]*b[5]+a[6]*b[8],a[0]*b[0]+a[3]*b[3]+a[6]*b[6]);
    t.ato_gap[i]=MeshGap(m,d,t.ato_geom[i]);
  }
  t.observed=true;
  double ats[2]{},ato[2]{}; std::vector<Point> support;
  for(int k=0;k<d->ncon;++k) {
    const auto& c=d->contact[k];
    const int g=c.geom1==t.floor?c.geom2:c.geom2==t.floor?c.geom1:-1;
    if(g<0) continue;
    mjtNum f[6]{}; mj_contactForce(m,d,k,f); const double fn=std::max(0.0,f[0]);
    for(int i=0;i<2;++i) { if(g==t.ats_geom[i]) ats[i]+=fn; if(g==t.ato_geom[i]) ato[i]+=fn; }
    const char* name=mj_id2name(m,mjOBJ_GEOM,g);
    if(fn>1 && name && (std::string(name).starts_with("rw_arc_")||std::string(name).starts_with("lw_arc_"))) support.push_back({c.pos[0],c.pos[1]});
  }
  for(int i=0;i<2;++i) {
    t.ats_force[i]+=dt/(0.025+dt)*(ats[i]-t.ats_force[i]);
    t.ato_force[i]+=dt/(0.025+dt)*(ato[i]-t.ato_force[i]);
  }
  const auto* com=d->subtree_com+3*t.torso;
  const auto* ar=d->xpos+3*t.ats_body[0]; const auto* al=d->xpos+3*t.ats_body[1];
  t.com_forward=ex*(com[0]-(ar[0]+al[0])*0.5)+ey*(com[1]-(ar[1]+al[1])*0.5);
  t.support_margin=HullMargin(support,{com[0],com[1]});
  t.chest_height=d->site_xpos[3*t.imu_site+2];
  // 实测构型给出的重力平衡前馈；只用于轮式支撑，不把胸高当作CoM。
  const double h=com[2]-(ar[2]+al[2])*0.5;
  const double eq=u.imu->GetPitch()-std::atan2(t.com_forward,std::max(0.2,h));
  t.equilibrium_pitch+=dt/(0.15+dt)*(Limit(eq,0.10)-t.equilibrium_pitch);
}
void StartGoals(UserData& u) {
  auto& t=u.transition;
  const double delta=std::remainder(t.phase[0]-t.phase[1],2*pi);
  const double left=t.phase[0]-delta, mean=0.5*(t.phase[0]+left);
  const double target=mean+std::remainder(-5*deg-mean,2*pi);
  t.phase_goal={target,target+t.phase[1]-left};
  t.chest_reference=t.chest_height;
  u.velocity_integral=0;
  t.goals_initialized=true; t.progress=0; t.guard_code=0;
  Enter(u,TransitionStage::Align);
}
bool CaptureFoot(UserData& u) {
  auto& t=u.transition;
  for(int j=0;j<10;++j) {
    t.captured_pose[j]=u.joints[j]->GetActualPosition();
    t.joint_target[j]=t.captured_pose[j];
    t.joint_integral[j]=u.command[j];
  }
  t.body_pitch_reference=u.imu->GetPitch(); t.foot_yaw=u.imu->GetYaw();
  t.initial_com_forward=t.com_forward;
  for(int axis=0;axis<3;++axis) t.foot_origin[axis]=0.5*(u.data->xpos[3*t.ats_body[0]+axis]+u.data->xpos[3*t.ats_body[1]+axis]);
  t.foot_ik=std::make_shared<FootIK>();
  std::array<int,8> ids{t.joint_id[0],t.joint_id[1],t.joint_id[2],t.joint_id[3],t.joint_id[5],t.joint_id[6],t.joint_id[7],t.joint_id[8]};
  if(!t.foot_ik->Initialize(u.model,u.data,t.torso,t.imu_site,ids)) return false;
  u.velocity_integral=0;
  t.progress=0;
  return true;
}
void FootTargets(UserData& u,double dt) {
  auto& t=u.transition;
  const double age=u.data->time-t.stage_started;
  t.foot_blend=Smooth(age/1.0);
  // 先进入支撑域靠后但有正裕量的位置，避免以支撑几何中心强迫NUT过载。
  const bool morphology_ok=std::abs(u.joints[4]->GetActualPosition()-stroke)<0.0004 &&
                           std::abs(u.joints[9]->GetActualPosition()-stroke)<0.0004;
  // 初触后允许动态平衡阶段的短暂ATO卸载；否则质心不前移、ATO不承载会互相等待。
  if(morphology_ok && t.ats_force[0]>15 && t.ats_force[1]>15 &&
      std::abs(u.command[4])<112 && std::abs(u.command[9])<112)
    t.progress=std::min(1.0,t.progress+dt/4.0);
  const double desired_com=t.initial_com_forward+(t.com_goal_forward-t.initial_com_forward)*Smooth(t.progress);
  std::array<double,3> com_goal=t.foot_origin;
  com_goal[0]+=desired_com*std::cos(t.foot_yaw);
  com_goal[1]+=desired_com*std::sin(t.foot_yaw);
  std::array<double,8> targets{};
  if(t.foot_ik && t.foot_ik->Step(u.data,dt,com_goal,t.chest_reference,targets)) {
    for(int i=0;i<8;++i) t.joint_target[i<4?i:i+1]=targets[i];
    if(t.guard_code==4 && t.foot_ik->diagnostics().updated) t.guard_code=0;
  } else { t.guard_code=4; } // 失败时保持上次有效关节参考，不使用未经约束的替代IK。
  // 在质心获得正裕量前保留倒立摆反馈，防止轻触地时提前撤掉平衡。
  const double h=std::max(0.2,u.data->subtree_com[3*t.torso+2]-t.foot_origin[2]);
  t.equilibrium_pitch=u.imu->GetPitch()-std::atan2(t.com_forward-desired_com,h);
  const double loaded=std::clamp(std::min(t.ato_force[0],t.ato_force[1])/15.0,0.0,1.0)*std::clamp(t.support_margin/0.008,0.0,1.0);
  t.ankle_blend+=Limit(loaded-t.ankle_blend,(loaded<t.ankle_blend?3.0:0.5)*dt);
}
}

const char* TransitionStageName(TransitionStage s) {
  switch(s) {
    case TransitionStage::Wheel:return "wheel_balance";
    case TransitionStage::Align:return "phase_alignment";
    case TransitionStage::Unfold:return "unfolding";
    case TransitionStage::Approach:return "contact_approach";
    case TransitionStage::LoadTransfer:return "load_transfer";
    case TransitionStage::Foot:return "foot_support";
    default:return "recovery_hold";
  }
}

void ConfigureTransition(UserData& u) {
  auto& t=u.transition;
  for(int j=0;j<10;++j) {
    t.joint_id[j]=u.model->actuator_trnid[2*u.actuator_ids[j]];
    t.qpos_id[j]=u.model->jnt_qposadr[t.joint_id[j]];
    t.dof_id[j]=u.model->jnt_dofadr[t.joint_id[j]];
  }
  t.torso=mj_name2id(u.model,mjOBJ_BODY,"torso");
  t.imu_site=mj_name2id(u.model,mjOBJ_SITE,"site_imu");
  t.floor=mj_name2id(u.model,mjOBJ_GEOM,"floor");
  if(t.torso<0 || t.imu_site<0 || t.floor<0)
    throw std::runtime_error("Transition requires torso, site_imu and floor");
  t.lqr_gain=LqrGains(u.model->body_subtreemass[t.torso]);
  for(double gain:t.lqr_gain) if(!std::isfinite(gain) || gain<=0 || gain>1000) throw std::runtime_error("Invalid transition LQR solution");
  std::cout<<"Transition LQR gains: "<<t.lqr_gain[0]<<","<<t.lqr_gain[1]<<","<<t.lqr_gain[2]<<","<<t.lqr_gain[3]<<"\n";
  for(int i=0;i<2;++i) {
    t.ats_body[i]=mj_name2id(u.model,mjOBJ_BODY,i?"LATS":"RATS");
    t.ato_body[i]=mj_name2id(u.model,mjOBJ_BODY,i?"LATO":"RATO");
    t.ats_geom[i]=mj_name2id(u.model,mjOBJ_GEOM,i?"lw_arc_LATS_00":"rw_arc_RATS_00");
    t.ato_geom[i]=mj_name2id(u.model,mjOBJ_GEOM,i?"lw_arc_LATO_00":"rw_arc_RATO_00");
    if(t.ats_body[i]<0||t.ato_body[i]<0||t.ats_geom[i]<0||t.ato_geom[i]<0) throw std::runtime_error("Transition requires ATS/ATO model bodies and collision meshes");
    const int mesh=u.model->geom_dataid[t.ato_geom[i]];
    if(u.model->geom_type[t.ato_geom[i]]!=mjGEOM_MESH || mesh<0 || mesh>=u.model->nmesh)
      throw std::runtime_error("Transition requires ATO mesh collision geometry");
    const int nut=t.joint_id[5*i+4];
    if(u.model->jnt_type[nut]!=mjJNT_SLIDE || std::abs(u.model->jnt_range[2*nut+1]-stroke)>1e-7)
      throw std::runtime_error("Transition calibration requires NUT upper stroke 0.040878 m");
  }
}

void UpdateTransition(UserData& u,double dt,double pitch_rate) {
  auto& t=u.transition;
  Observe(u,dt,pitch_rate);
  if(t.requested && t.stage==TransitionStage::Wheel) { t.requested=false; StartGoals(u); }
  if(t.stage==TransitionStage::Wheel) return;
  if(t.stage==TransitionStage::Hold) {
    t.common_velocity_target=0; t.differential_torque=0; t.nut_velocity=0;
    const double loaded=std::clamp(std::min(t.ato_force[0],t.ato_force[1])/15.0,0.0,1.0)*std::clamp(t.support_margin/0.008,0.0,1.0);
    if(loaded<t.ankle_blend) t.ankle_blend=std::max(loaded,t.ankle_blend-3.0*dt);
    return;
  }
  const double now=u.data->time;
  const double er=t.phase[0]-t.phase_goal[0],el=t.phase[1]-t.phase_goal[1];
  const double phase_error=0.5*(er+el),delta=er-el;
  const double speed=std::max(std::abs(t.phase_rate[0]),std::abs(t.phase_rate[1]));
  if(t.stage==TransitionStage::Approach)
    t.approach_contact_detected=t.approach_contact_detected || t.ato_force[0]>1 || t.ato_force[1]>1;
  const double limit=t.stage==TransitionStage::Approach?0.012:0.07;
  double requested=Limit(-(t.lqr_gain[0]/t.lqr_gain[1])*u.params.wheel_radius*phase_error,limit);
  // 展开后滚向触地相位：远处允许较快接近，临近地面降低速度上限。
  // 正的爬行参考避免相位P趋零后再次停在离散轮面上；只在ATO悬空时启用。
  if(t.stage==TransitionStage::Approach && phase_error<0 &&
     !t.approach_contact_detected && std::min(t.ato_gap[0],t.ato_gap[1])>0) {
    const double gap=std::min(t.ato_gap[0],t.ato_gap[1]);
    const double approach_limit=0.004+0.008*Smooth((gap-0.001)/0.002);
    requested=std::min(std::max(requested,0.004),approach_limit);
  }
  t.common_velocity_target=t.previous_common_velocity+Limit(requested-t.previous_common_velocity,0.025*dt);
  t.previous_common_velocity=t.common_velocity_target;
  t.differential_torque=Limit(7.0*delta+2.0*(t.phase_rate[0]-t.phase_rate[1]),3.0);

  if(t.stage==TransitionStage::Align) {
    const bool ready=std::abs(er)<1.8*deg && std::abs(el)<1.8*deg && std::abs(delta)<0.5*deg &&
                     speed<0.8*deg && t.ats_force[0]>30 && t.ats_force[1]>30 &&
                     std::abs(u.imu->GetRoll())<3*deg && std::abs(u.imu->GetPitch()-t.equilibrium_pitch)<2*deg;
    t.settled=ready?t.settled+dt:0;
    if(t.settled>0.5) {
      // 1度离散圆弧存在半度平面驻留。接受安全窗内的稳态，不强迫精确-5度。
      t.phase_goal[0]+=phase_error; t.phase_goal[1]+=phase_error;
      t.previous_common_velocity=0; t.common_velocity_target=0;
      u.velocity_integral=0;
      Enter(u,TransitionStage::Unfold); t.progress=0;
    }
  } else if(t.stage==TransitionStage::Unfold) {
    const double pr=std::remainder(t.phase[0],2*pi),pl=std::remainder(t.phase[1],2*pi);
    const double prediction=0.15*speed;
    const bool clear=pr>-7*deg+prediction && pr<-3*deg-prediction && pl>-7*deg+prediction && pl<-3*deg-prediction &&
                     t.ats_force[0]>15 && t.ats_force[1]>15 &&
                     t.ato_gap[0]>0.002 && t.ato_gap[1]>0.002 && t.ato_force[0]<2 && t.ato_force[1]<2;
    t.guard_code=clear?0:1;
    if(t.ato_force[0]>3 || t.ato_force[1]>3) {
      t.guard_code=3; Enter(u,TransitionStage::Hold); return;
    }
    if(clear) t.progress=std::min(1.0,t.progress+dt/8.0);
    t.nut_target=stroke*Smooth(t.progress);
    t.nut_velocity=clear?stroke*SmoothRate(t.progress)/8.0:0;
    const bool done=t.progress>=1 && std::abs(u.joints[4]->GetActualPosition()-stroke)<0.00015 &&
      std::abs(u.joints[9]->GetActualPosition()-stroke)<0.00015 &&
      std::abs(t.alpha[0]-pi/3)<0.7*deg && std::abs(t.alpha[1]-pi/3)<0.7*deg &&
      std::abs(u.joints[4]->GetActualVelocity())<0.0003 && std::abs(u.joints[9]->GetActualVelocity())<0.0003;
    t.settled=done?t.settled+dt:0;
    if(t.settled>0.4) {
      t.phase_goal[0]=std::round(t.phase_goal[0]/(2*pi))*2*pi;
      t.phase_goal[1]=std::round(t.phase_goal[1]/(2*pi))*2*pi;
      t.guard_code=0;
      // 保留已经建立的公共力矩偏置，避免展开末端清零后重新等待积分积累。
      t.approach_contact_detected=false;
      Enter(u,TransitionStage::Approach);
    }
  } else if(t.stage==TransitionStage::Approach) {
    t.nut_velocity=0;
    // 首触即减缓接近和差动动作，不等待第二侧才响应。
    // 首触锁存：短暂卸载不能重新启用快速接近。
    if(t.approach_contact_detected) {
      t.common_velocity_target=Limit(t.common_velocity_target,0.001);
      t.previous_common_velocity=t.common_velocity_target;
      t.differential_torque=0; // 已接触时不再使用双圆轮的普通差动追相。
      if(std::abs(delta)>0.8*deg) { t.guard_code=3; Enter(u,TransitionStage::Hold); return; }
    }
    const bool both=t.ato_force[0]>0.5 && t.ato_force[1]>0.5 && t.ato_gap[0]<0.0005 && t.ato_gap[1]<0.0005 &&
                    t.ats_force[0]>15 && t.ats_force[1]>15;
    t.settled=both?t.settled+dt:0;
    if(t.settled>0.15) {
      if(CaptureFoot(u)) Enter(u,TransitionStage::LoadTransfer);
      else { t.guard_code=4; Enter(u,TransitionStage::Hold); }
    }
  } else if(t.stage==TransitionStage::LoadTransfer || t.stage==TransitionStage::Foot) {
    t.common_velocity_target=0; t.differential_torque=0; t.nut_velocity=0;
    FootTargets(u,dt);
    const bool stable=t.ats_force[0]>5 && t.ats_force[1]>5 && t.ato_force[0]>5 && t.ato_force[1]>5 &&
      t.support_margin>0.008 && std::abs(t.com_forward-t.com_goal_forward)<0.006 &&
      std::abs(t.chest_height-t.chest_reference)<0.012 && std::abs(u.imu->GetRoll())<3*deg && speed<0.5*deg &&
      std::abs(t.alpha[0]-pi/3)<deg && std::abs(t.alpha[1]-pi/3)<deg &&
      std::abs(u.joints[4]->GetActualPosition()-stroke)<0.0003 && std::abs(u.joints[9]->GetActualPosition()-stroke)<0.0003;
    t.settled=stable?t.settled+dt:0;
    if(t.stage==TransitionStage::LoadTransfer && t.settled>2.0) {
      // 保留轨迹时钟/接触参考，防止进入Foot时重置质心轨迹。
      t.stage=TransitionStage::Foot;
      std::cout<<"[transition "<<now<<"s] foot_support; margin="<<t.support_margin<<" m\n";
    }
    const bool degraded=t.ats_force[0]<3 || t.ats_force[1]<3 ||
      std::abs(u.joints[4]->GetActualPosition()-stroke)>0.001 ||
      std::abs(u.joints[9]->GetActualPosition()-stroke)>0.001 || t.guard_code==4 ||
      (t.stage==TransitionStage::Foot && (t.ato_force[0]<2 || t.ato_force[1]<2 || t.support_margin<0.002));
    t.unstable_time=degraded?t.unstable_time+dt:0;
    if(t.unstable_time>0.3) { t.guard_code=5; Enter(u,TransitionStage::Hold); }
  }
  if(now-t.stage_started>100 && t.stage!=TransitionStage::Foot && t.stage!=TransitionStage::Hold) {
    t.guard_code=2; Enter(u,TransitionStage::Hold);
  }
}

double TransitionJointTorque(UserData& u,std::size_t j,double normal,double dt) {
  auto& t=u.transition; const int axis=static_cast<int>(j%5);
  if(axis==4 && t.stage!=TransitionStage::Wheel) {
    const int i=static_cast<int>(j/5); const double error=t.nut_target-u.joints[j]->GetActualPosition();
    if(t.stage==TransitionStage::Approach || t.stage==TransitionStage::LoadTransfer || t.stage==TransitionStage::Foot || t.stage==TransitionStage::Hold) {
      double force=0,residual=0;
      if(ComputeShapeHoldingForce(u.model,u.data,i,force,residual)) {
        force=Limit(force,100);
        if(!t.feedforward_started[i]) {
          t.nut_feedforward[i]=force;
          t.nut_integral[i]-=force; // 无扰接入，随后随实测承载变化提供前馈。
          t.feedforward_started[i]=true;
        } else t.nut_feedforward[i]+=dt/(0.03+dt)*(force-t.nut_feedforward[i]);
      }
    }
    const double pd=40000*error+300*(t.nut_velocity-u.joints[j]->GetActualVelocity());
    const double candidate=Limit(t.nut_integral[i]+40000*error*dt,110);
    if(std::abs(pd+candidate+t.nut_feedforward[i])<115 || error*(pd+candidate+t.nut_feedforward[i])<0) t.nut_integral[i]=candidate;
    const double force=Limit(pd+t.nut_integral[i]+t.nut_feedforward[i],115);
    const double limited=t.nut_last_command[i]+Limit(force-t.nut_last_command[i],2000*dt);
    t.nut_last_command[i]=limited; return limited;
  }
  if(t.foot_blend>0 && axis!=4) {
    constexpr double kp[4]{800,1200,1600,500},kd[4]{30,40,50,25},lim[4]{100,100,200,40};
    const double error=t.joint_target[j]-u.joints[j]->GetActualPosition();
    const double pd=kp[axis]*error-kd[axis]*u.joints[j]->GetActualVelocity();
    const double candidate=Limit(t.joint_integral[j]+120*error*dt,lim[axis]*0.8);
    if(std::abs(pd+candidate)<lim[axis] || error*(pd+candidate)<0) t.joint_integral[j]=candidate;
    const double foot=Limit(pd+t.joint_integral[j],lim[axis]);
    const double blend=axis==3?t.ankle_blend:t.foot_blend;
    return (1-blend)*normal+blend*foot;
  }
  return normal;
}
