import numpy as np
import os
import matplotlib.pyplot as plt

SIM_LENGTH = 20000  # match test outer loop steps: int(SIM_DURATION/0.001) for default accel SIM_DURATION≈36s
START_IDX = 0    # only plot data from this sim step onward (set to 0 to disable)

DATA_DICT = {
    "qpin": 37,
    "qjnt_d": 30,
    "vpin" : 36,
    "vpin_d": 36,
    "uf": 20,
    "h_cur": 18,
    "h_cur_d": 18,
    "hd": 18,
    "hd_dot": 18,
    "hd_ddot": 18,
    "x_sc_des": 1,
    "y_sc_des": 1,
    "x_sc_cur": 1,
    "y_sc_cur": 1,
    "x_eq": 1,
    "t": 1,
    "L_GRF": 1,
    "R_GRF": 1,
    "u_x": 1,
    "u_y": 1,
    "u_x_act": 1,
    "u_y_act": 1,
    "x_pre_est": 1,
    "v_pre_est": 1,
    "alip_L_cur": 1,
    "alip_L_pre_est": 1,
    "hlip_x_cur": 1,
    "hlip_v_cur": 1,
    "com_world_x": 1,
    "stance_world_x": 1,
    "com_world_vx": 1,
    "stance_world_vx": 1,
    "plate_world_x": 1,
    "plate_world_vx": 1,
    "v_com_minus_stance": 1,
    "v_com_minus_plate": 1,
    "v_stance_minus_plate": 1,
    "proj_plate_x": 1,
    "proj_plate_vx": 1,
    "proj_plate_angle": 1,
    "com_plate_vx": 1,
    "com_plate_vx_cmd": 1,
    "com_plate_vx_err": 1,
    "hlip_K": 2,
    "hlip_z_star": 2,
    "tau_ax": 1,
    "tau_ay": 1,
    "tau_y_gravity": 1,
    "tau_y_bias": 1,
    "tau_y_ff": 1,
    "tau_y_P": 1,
    "tau_y_D": 1,
    "tau_y_PD": 1,
    "foot_state": 1,
    "uf_stand": 20,
    "uf_fb": 20,
    "uf_ff": 20,
    "ada_x_obsever_state":          2,   # = 2 dim([x_sc, vx_sc])    # Adaptive controller's information for x_sc
    "ada_x_compensator_state":     55,   # = 210 = 0.5 * comp_n * (comp_n + 1)
    "ada_x_regressor_state":       12,   # = 22 = comp_n + #of observer state ?
    "ada_x_compensator_coef":      10,   # = 20 = comp_n
    "ada_x_covariance_matrix_norm": 1,
    "ada_y_tau": 1,
    "ada_y_obsever_state":          2, # Adaptive controller's information for y_sc
    "ada_y_compensator_state":     55,
    "ada_y_regressor_state":       12,
    "ada_y_compensator_coef":      10,
    "ada_y_covariance_matrix_norm": 1,
    "ada_x_tau": 1,
    "drs_des": 6,
    "drs_act": 6,
    "plate_ax_cmd": 1,  # commanded plate sagittal a_x from ctrl_drs FD [m/s^2]
    "GRF_mlc": 4,
    "pre_impact": 1,
    "double_support": 1,
    "output_x_des": 2,
    "output_x_act": 2,
    "output_y_des": 2,
    "output_y_act": 2,
    "plate_imu_acc": 3,
    "plate_imu_acc_stance": 3,
    "torso_imu_acc": 3,
    "torso_imu_quat": 4,
    "torso_imu_acc_stance": 3,
    "torso_imu_R_stance": 9,
    "geff_plate": 2,
    "geff_torso_projected": 2,
    "torso_fk_angles": 4,
    "torso_proj_imu_acc": 3,
    "torso_proj_fk_foot_over_torso_acc": 3,
    "torso_proj_result_acc": 3,
    "torso_proj_result_lpf_acc": 3,
    "torso_proj_result_slope_acc": 3,
    "x_pre": 1,
    "vx_pre": 1,
    "t_pre": 1,
    "s_pre": 1,
    "foot_pre": 1,
    "x_pd": 1,
    "x_ff": 1,
    "x_ddot_ff": 1,
    "x_ddot_biased": 1,
    "x_sc_ddot_meas": 1,
    "y_sc_ddot_meas": 1,
    "z_sc_ddot_meas": 1,
    "x_sc_ddot_des_hlip": 1,
}

def ezplot_joint(a,b,c,name1,idx1,name2,idx2,title):
    plt.subplot(a,b,c)
    plt.plot(dict_array["t"][:,0],dict_array[name1][:,idx1]*180/np.pi,'--')
    plt.plot(dict_array["t"][:,0],dict_array[name2][:,idx2]*180/np.pi,'-')
    plt.title(title)
    plt.grid(True,'both','both')
    
def ezplot_joint1(a,b,c,name1,idx1,title):
    plt.subplot(a,b,c)
    plt.plot(dict_array["t"][:,0],dict_array[name1][:,idx1],'-')
    plt.title(title)
    plt.grid(True,'both','both')

def ezplot_joint2(a,b,c,name1,idx1,linestyle):
    plt.subplot(a,b,c)
    plt.plot(dict_array["t"][:,0],dict_array[name1][:,idx1],linestyle)
    plt.grid(True,'both','both')
    
def ezplot_joint2_deg(a,b,c,name1,idx1,linestyle):
    plt.subplot(a,b,c)
    plt.plot(dict_array["t"][:,0],dict_array[name1][:,idx1]*180/np.pi,linestyle)
    plt.grid(True,'both','both')

def _plate_ax_second_diff(t_vec, x_vec):
    """Sagittal plate a_x ≈ (x_k - 2 x_{k-1} + x_{k-2}) / dt^2 (matches digit_controller DRS diff)."""
    t_vec = np.asarray(t_vec, dtype=float).ravel()
    x_vec = np.asarray(x_vec, dtype=float).ravel()
    n = min(len(t_vec), len(x_vec))
    out = np.full(n, np.nan, dtype=float)
    for k in range(2, n):
        t_now = t_vec[k]
        t_km1 = t_vec[k - 1]
        t_km2 = t_vec[k - 2]
        dt1 = t_now - t_km1
        dt2 = t_km1 - t_km2
        if dt1 > 1e-12 and abs(dt1 - dt2) < 1e-6:
            out[k] = (x_vec[k] - 2.0 * x_vec[k - 1] + x_vec[k - 2]) / (dt1 * dt1)
    return out

#####################
# plot DRS tracking #
#####################
def plot_DRS_tracking():
    plt.figure("DRS tracking")
    
    ezplot_joint2(3,4,1,"drs_des",0,'--')
    ezplot_joint2(3,4,1,"drs_act",0,'-')
    plt.ylabel('x [m]')
    plt.ylim([-0.10, 0.10])
    
    ezplot_joint2(3,4,2,"drs_des",1,'--')
    ezplot_joint2(3,4,2,"drs_act",1,'-')
    plt.ylabel('y [m]')
    plt.ylim([-0.10, 0.10])
    
    ezplot_joint2(3,4,3,"drs_des",2,'--')
    ezplot_joint2(3,4,3,"drs_act",2,'-')
    plt.ylabel('z [m]')
    plt.ylim([0.80, 1.00])
    
    ezplot_joint2_deg(3,4,4,"drs_des",3,'--')
    ezplot_joint2_deg(3,4,4,"drs_act",3,'-')
    plt.ylabel('roll [deg]')
    plt.ylim([-15, 15])
    
    ezplot_joint2_deg(3,4,5,"drs_des",4,'--')
    ezplot_joint2_deg(3,4,5,"drs_act",4,'-')
    plt.ylabel('pitch [deg]')
    plt.ylim([-15, 15])
    
    ezplot_joint2_deg(3,4,6,"drs_des",5,'--')
    ezplot_joint2_deg(3,4,6,"drs_act",5,'-')
    plt.ylabel('yaw [deg]')
    plt.ylim([-15, 15])

    t = dict_array["t"][:, 0]
    ax_cmd = dict_array["plate_ax_cmd"][:, 0]
    ax_act = _plate_ax_second_diff(t, dict_array["drs_act"][:, 0])
    plt.subplot2grid((3, 4), (2, 0), colspan=4)
    plt.plot(t, ax_cmd, "--", label=r"$a_x$ commanded (finite-diff of $x_{\mathrm{des}}$)")
    plt.plot(t, ax_act, "-", label=r"$a_x$ actual (FD of drs_act)")
    plt.ylabel(r"$a_x$ [m/s$^2$]")
    plt.xlabel("t [sec]")
    plt.grid(True, "both", "both")
    plt.legend(loc="best")
    

###############
# plot torque #
###############
def plot_left_leg_torque():
    plt.figure("Left leg torque")
    
    ezplot_joint1(3,4,1,"uf",0,"left hip roll (Nm)")
    ezplot_joint1(3,4,2,"uf",1,"left hip yaw (Nm)")
    ezplot_joint1(3,4,3,"uf",2,"left hip pitch (Nm)")
    ezplot_joint1(3,4,4,"uf",3,"left knee (Nm)")
    
    ezplot_joint1(3,4,8,"uf",4,"left toe A (Nm)")
    ezplot_joint1(3,4,9,"uf",5,"left toe B (Nm)")
    
def plot_left_leg_torque_compare():
    plt.figure("Left leg torque compare")
    
    ezplot_joint1(2,3,1,"uf",0,"left hip roll (Nm)")
    ezplot_joint2(2,3,1,"uf_stand",0,"k--")
    ezplot_joint2(2,3,1,"uf_fb",0,"r-")
    ezplot_joint2(2,3,1,"uf_ff",0,"b-")
    plt.ylim([-75,75])
    
    ezplot_joint1(2,3,2,"uf",1,"left hip yaw (Nm)")
    ezplot_joint2(2,3,2,"uf_stand",1,"k--")
    ezplot_joint2(2,3,2,"uf_fb",1,"r-")
    ezplot_joint2(2,3,2,"uf_ff",1,"b-")
    plt.ylim([-5,5])
    
    ezplot_joint1(2,3,3,"uf",2,"left hip pitch (Nm)")
    ezplot_joint2(2,3,3,"uf_stand",2,"k--")
    ezplot_joint2(2,3,3,"uf_fb",2,"r-")
    ezplot_joint2(2,3,3,"uf_ff",2,"b-")
    plt.ylim([-15,15])
    
    ezplot_joint1(2,3,4,"uf",3,"left knee (Nm)")
    ezplot_joint2(2,3,4,"uf_stand",3,"k--")
    ezplot_joint2(2,3,4,"uf_fb",3,"r-")
    ezplot_joint2(2,3,4,"uf_ff",3,"b-")
    plt.ylim([-250, 250])
    
    
    ezplot_joint1(2,3,5,"uf",4,"left toe A (Nm)")
    ezplot_joint2(2,3,5,"uf_stand",4,"k--")
    ezplot_joint2(2,3,5,"uf_fb",4,"r-")
    ezplot_joint2(2,3,5,"uf_ff",4,"b-")
    plt.ylim([-15,15])
    
    ezplot_joint1(2,3,6,"uf",5,"left toe B (Nm)")
    ezplot_joint2(2,3,6,"uf_stand",5,"k--")
    ezplot_joint2(2,3,6,"uf_fb",5,"r-")
    ezplot_joint2(2,3,6,"uf_ff",5,"b-")
    plt.ylim([-15,15])
    plt.legend(["uf","uf stand","uf fb","uf_ff"])
        
def plot_right_leg_torque():
    plt.figure("Right leg torque")
    
    ezplot_joint1(3,4,1,"uf",10,"right hip roll (Nm)")
    ezplot_joint1(3,4,2,"uf",11,"right hip yaw (Nm)")
    ezplot_joint1(3,4,3,"uf",12,"right hip pitch (Nm)")
    ezplot_joint1(3,4,4,"uf",13,"right knee (Nm)")
    
    ezplot_joint1(3,4,8,"uf",14,"right toe A (Nm)")
    ezplot_joint1(3,4,9,"uf",15,"right toe B (Nm)")

def plot_right_leg_torque_compare():
    plt.figure("Right leg torque compare")
    
    ezplot_joint1(2,3,1,"uf",10,"right hip roll (Nm)")
    ezplot_joint2(2,3,1,"uf_stand",10,"--")
    ezplot_joint2(2,3,1,"uf_fb",10,"r-")
    ezplot_joint2(2,3,1,"uf_ff",10,"b-")
    plt.ylim([-75,75])
    
    ezplot_joint1(2,3,2,"uf",11,"right hip yaw (Nm)")
    ezplot_joint2(2,3,2,"uf_stand",11,"--")
    ezplot_joint2(2,3,2,"uf_fb",11,"r-")
    ezplot_joint2(2,3,2,"uf_ff",11,"b-")
    plt.ylim([-5,5])
    
    ezplot_joint1(2,3,3,"uf",12,"right hip pitch (Nm)")
    ezplot_joint2(2,3,3,"uf_stand",12,"--")
    ezplot_joint2(2,3,3,"uf_fb",12,"r-")
    ezplot_joint2(2,3,3,"uf_ff",12,"b-")
    plt.ylim([-15,15])
    
    ezplot_joint1(2,3,4,"uf",13,"right knee (Nm)")
    ezplot_joint2(2,3,4,"uf_stand",13,"--")
    ezplot_joint2(2,3,4,"uf_fb",13,"r-")
    ezplot_joint2(2,3,4,"uf_ff",13,"b-")
    plt.ylim([-250, 250])
    
    
    ezplot_joint1(2,3,5,"uf",14,"right toe A (Nm)")
    ezplot_joint2(2,3,5,"uf_stand",14,"--")
    ezplot_joint2(2,3,5,"uf_fb",14,"r-")
    ezplot_joint2(2,3,5,"uf_ff",14,"b-")
    plt.ylim([-15,15])
    
    ezplot_joint1(2,3,6,"uf",15,"right toe B (Nm)")
    ezplot_joint2(2,3,6,"uf_stand",15,"--")
    ezplot_joint2(2,3,6,"uf_fb",15,"r-")
    ezplot_joint2(2,3,6,"uf_ff",15,"b-")
    plt.ylim([-15,15])
    plt.legend(["uf","uf stand","uf fb","uf_ff"])

#######################
# task space tracking #
#######################
    
def plot_arm_task_space_tracking():
    plt.figure("Arm task space tracking")
    
    ezplot_joint(2,4,1,"hd",10,"h_cur",10,"left shoulder roll (deg)")
    ezplot_joint(2,4,2,"hd",11,"h_cur",11,"left shoulder pitch (deg)")
    ezplot_joint(2,4,3,"hd",12,"h_cur",12,"left shoulder yaw (deg)")
    ezplot_joint(2,4,4,"hd",13,"h_cur",13,"left elbow (deg)")
    
    ezplot_joint(2,4,5,"hd",14,"h_cur",14,"right shoulder roll (deg)")
    ezplot_joint(2,4,6,"hd",15,"h_cur",15,"right shoulder pitch (deg)")
    ezplot_joint(2,4,7,"hd",16,"h_cur",16,"right shoulder yaw (deg)")
    ezplot_joint(2,4,8,"hd",17,"h_cur",17,"right elbow (deg)")


# Task-space DoF excluding COM (index 0, x/y/z_sc channels) and swing foot (indices 4–9): base + arms only.
_TASK_SPACE_BASE_ARM_INDICES = (1, 2, 3, 10, 11, 12, 13, 14, 15, 16, 17)
_TASK_SPACE_BASE_ARM_LABELS = [
    "base roll", "base pitch", "base yaw",
    "L sh roll", "L sh pitch", "L sh yaw", "L elbow",
    "R sh roll", "R sh pitch", "R sh yaw", "R elbow",
]


def _plot_task_space_tracking_base_arms_only(num, suptitle):
    """Desired vs actual for base orientation and arms (no COM, no swing foot)."""
    if "hd" not in dict_array or "h_cur" not in dict_array:
        return
    t = dict_array["t"][:, 0]
    hd = dict_array["hd"]
    hc = dict_array["h_cur"]
    subplots_kw = dict(nrows=3, ncols=4, figsize=(16, 8))
    if num is not None:
        subplots_kw["num"] = num
    fig, axes = plt.subplots(**subplots_kw)
    fig.suptitle(suptitle, fontsize=12)
    axes_flat = axes.flatten()
    for k, i in enumerate(_TASK_SPACE_BASE_ARM_INDICES):
        ax = axes_flat[k]
        ax.plot(t, hd[:, i] * 180 / np.pi, "--", label="des")
        ax.plot(t, hc[:, i] * 180 / np.pi, "-", label="act")
        ax.set_ylabel("deg")
        ax.set_title(_TASK_SPACE_BASE_ARM_LABELS[k], fontsize=9)
        ax.grid(True, "both", "both")
    axes_flat[-1].set_visible(False)
    if axes_flat.size:
        axes_flat[0].legend(loc="upper right", fontsize=7)
    plt.tight_layout()


def plot_task_space_tracking():
    _plot_task_space_tracking_base_arms_only("Task space tracking", "Task space tracking")


def plot_COM_tracking():
    """COM movement tracking: desired vs actual x_sc only."""
    if "x_sc_des" not in dict_array or "x_sc_cur" not in dict_array:
        return
    plt.figure("COM tracking")
    t = dict_array["t"][:, 0]
    plt.plot(t, dict_array["x_sc_des"][:, 0], "--", label="des")
    plt.plot(t, dict_array["x_sc_cur"][:, 0], "-", label="actual")
    plt.xlabel("t (s)")
    plt.ylabel("m")
    plt.title(r"$x_{sc}$ (m)")
    plt.ylim(-0.1, 0.1)
    plt.legend(loc="upper right", fontsize=8)
    plt.grid(True, "both", "both")
    plt.tight_layout()


def plot_x_pd_ref_vs_x_sc():
    """PD position reference (x_pd = x_d + x_eq) vs actual sagittal COM x_sc."""
    if "x_pd" not in dict_array or "x_sc_cur" not in dict_array:
        return
    plt.figure(r"$x_{pd,ref}$ vs $x_{sc}$", figsize=(8, 4))
    t = dict_array["t"][:, 0]
    plt.plot(t, dict_array["x_pd"][:, 0], "--", label=r"$x_{pd,ref}$ (PD ref)")
    plt.plot(t, dict_array["x_sc_cur"][:, 0], "-", label=r"$x_{sc}$ (actual)")
    plt.xlabel("t (s)")
    plt.ylabel("m")
    plt.title(r"Sagittal COM: PD reference vs actual")
    plt.legend(loc="upper right", fontsize=9)
    plt.grid(True, "both", "both")
    plt.tight_layout()
    plt.savefig("data/x_pd_ref_vs_x_sc.png", dpi=600)


def plot_COM_x_des_vs_actual_plus_xeq():
    """Sagittal COM: desired x_sc vs actual COM x plus biased-HLIP equilibrium x_eq."""
    if "x_sc_des" not in dict_array or "x_sc_cur" not in dict_array:
        return
    if "x_eq" not in dict_array:
        return
    plt.figure("COM x: des vs (actual + x_eq)")
    t = dict_array["t"][:, 0]
    x_eq = dict_array["x_eq"][:, 0]
    plt.plot(t, dict_array["x_sc_des"][:, 0], "--", label=r"$x_{sc}^{des}$")
    plt.plot(t, dict_array["x_sc_cur"][:, 0] + x_eq, "-", label=r"$x_{sc} + x_{eq}$")
    plt.xlabel("t (s)")
    plt.ylabel("m")
    plt.title(r"Sagittal COM: desired vs $x_{sc} + x_{eq}$")
    plt.ylim(-0.1, 0.1)
    plt.legend(loc="upper right", fontsize=8)
    plt.grid(True, "both", "both")
    plt.tight_layout()


def plot_COM_error():
    """COM error: desired - actual for x_sc, y_sc, z_sc."""
    if "x_sc_des" not in dict_array or "x_sc_cur" not in dict_array:
        return
    plt.figure("COM error")
    t = dict_array["t"][:, 0]
    # x_sc error
    plt.subplot(1, 3, 1)
    plt.plot(t, dict_array["x_sc_des"][:, 0] - dict_array["x_sc_cur"][:, 0], '-')
    plt.title("x_{sc} error (m)")
    plt.ylabel("des - actual")
    plt.ylim(-0.1, 0.1)
    plt.grid(True, 'both', 'both')
    # y_sc error
    plt.subplot(1, 3, 2)
    plt.plot(t, dict_array["y_sc_des"][:, 0] - dict_array["y_sc_cur"][:, 0], '-')
    plt.title("y_{sc} error (m)")
    plt.ylabel("des - actual")
    plt.ylim(-0.1, 0.1)
    plt.grid(True, 'both', 'both')
    # z_sc error (from task space hd/h_cur index 0)
    plt.subplot(1, 3, 3)
    if "hd" in dict_array and "h_cur" in dict_array:
        plt.plot(t, dict_array["hd"][:, 0] - dict_array["h_cur"][:, 0], '-')
    plt.title("z_{sc} error (m)")
    plt.ylabel("des - actual")
    plt.ylim(-0.1, 0.1)
    plt.grid(True, 'both', 'both')
    plt.tight_layout()


def plot_com_world_x_and_stance_world_x():
    """Plot CoM x and stance-foot x in world frame.

    Prefer true world-frame logs if available:
      - com_world_x
      - stance_world_x

    Otherwise reconstruct stance-foot world x by accumulating realized step lengths at touchdown:
      x_stance_world[k+1] = x_stance_world[k] + u_x_act at foot_state switch.

    Then:
      x_com_world = x_stance_world + x_sc_cur
      (since x_sc_cur is CoM relative to stance foot).
    """
    if "t" not in dict_array:
        return
    t = dict_array["t"][:, 0]

    have_true_world = ("com_world_x" in dict_array) and ("stance_world_x" in dict_array)
    if have_true_world and np.any(dict_array["com_world_x"]) and np.any(dict_array["stance_world_x"]):
        x_com_world = dict_array["com_world_x"][:, 0].astype(float)
        x_stance_world = dict_array["stance_world_x"][:, 0].astype(float)
        x_sc = dict_array["x_sc_cur"][:, 0].astype(float) if "x_sc_cur" in dict_array else None
    else:
        req = ("x_sc_cur", "u_x_act", "foot_state")
        if not all(k in dict_array for k in req):
            return

        x_sc = dict_array["x_sc_cur"][:, 0]
        u_x_act = dict_array["u_x_act"][:, 0]
        foot = dict_array["foot_state"][:, 0]

        n = t.shape[0]
        if n == 0:
            return

        x_stance_world = np.zeros(n, dtype=float)
        for i in range(1, n):
            x_stance_world[i] = x_stance_world[i - 1]
            if foot[i] != foot[i - 1]:
                x_stance_world[i] += float(u_x_act[i])

        x_com_world = x_stance_world + x_sc

    plt.figure("World x: CoM and stance foot", figsize=(9, 6))
    ax1 = plt.subplot(2, 1, 1)
    label_suffix = "(logged)" if have_true_world and np.any(dict_array["com_world_x"]) else "(reconstructed)"
    ax1.plot(t, x_com_world, "-", label=rf"$x_{{com}}^{{world}}$ {label_suffix}")
    ax1.plot(t, x_stance_world, "--", label=rf"$x_{{stance}}^{{world}}$ {label_suffix}")
    ax1.set_ylabel("x (m)")
    ax1.legend(loc="upper left", fontsize=8)
    ax1.grid(True, "both", "both")
    ax1.set_title("World-frame positions")

    ax2 = plt.subplot(2, 1, 2)
    ax2.plot(t, x_com_world - x_stance_world, "-", color="C3", label=r"$x_{com}^{world} - x_{stance}^{world}$")
    if x_sc is not None:
        ax2.plot(t, x_sc, "--", color="C0", alpha=0.7, label=r"$x_{sc}$")
    ax2.set_xlabel("t (s)")
    ax2.set_ylabel("difference (m)")
    ax2.legend(loc="upper left", fontsize=8)
    ax2.grid(True, "both", "both")
    plt.tight_layout()


def plot_velocity_frame_diagnostics():
    """Compare sagittal velocity measured in stance-foot, plate, and slip frames."""
    req = (
        "com_world_vx",
        "stance_world_vx",
        "plate_world_vx",
        "v_com_minus_stance",
        "v_com_minus_plate",
        "v_stance_minus_plate",
    )
    if not all(k in dict_array for k in req):
        return

    t = dict_array["t"][:, 0]
    v_com = dict_array["com_world_vx"][:, 0]
    v_stance = dict_array["stance_world_vx"][:, 0]
    v_plate = dict_array["plate_world_vx"][:, 0]
    v_com_stance = dict_array["v_com_minus_stance"][:, 0]
    v_com_plate = dict_array["v_com_minus_plate"][:, 0]
    v_stance_plate = dict_array["v_stance_minus_plate"][:, 0]

    plt.figure("Velocity frame diagnostics", figsize=(9, 8))

    ax1 = plt.subplot(3, 1, 1)
    ax1.plot(t, v_com, "-", label=r"$v_{com}^{world}$")
    ax1.plot(t, v_stance, "--", label=r"$v_{stance}^{world}$")
    ax1.plot(t, v_plate, ":", label=r"$v_{plate}^{world}$")
    ax1.set_ylabel("world vx (m/s)")
    ax1.legend(loc="upper right", fontsize=8)
    ax1.grid(True, "both", "both")

    ax2 = plt.subplot(3, 1, 2)
    ax2.plot(t, v_com_stance, "-", label=r"$v_{com} - v_{stance}$ (HLIP input frame)")
    ax2.plot(t, v_com_plate, "--", label=r"$v_{com} - v_{plate}$ (plate frame)")
    if "hlip_v_cur" in dict_array:
        ax2.plot(t, dict_array["hlip_v_cur"][:, 0], ":", color="C4", label="hlip_v_cur")
    ax2.set_ylabel("relative vx (m/s)")
    ax2.legend(loc="upper right", fontsize=8)
    ax2.grid(True, "both", "both")

    ax3 = plt.subplot(3, 1, 3)
    ax3.plot(t, v_stance_plate, "-", color="C3", label=r"$v_{stance} - v_{plate}$")
    ax3.axhline(0.0, color="k", linewidth=0.8, linestyle="--")
    ax3.set_xlabel("t (sec)")
    ax3.set_ylabel("foot slip vx (m/s)")
    ax3.legend(loc="upper right", fontsize=8)
    ax3.grid(True, "both", "both")

    plt.tight_layout()


def plot_velocity_error_from_nominal():
    """Controller input velocity along physical plate x-axis vs nominal command."""
    req = ("t", "com_plate_vx_cmd")
    if not all(k in dict_array for k in req):
        return

    t = dict_array["t"][:, 0]
    if os.path.exists("data/com_plate_vx.dat"):
        v_com = dict_array["com_plate_vx"][:, 0]
    else:
        # Backward compatibility for logs created before the generic velocity channel.
        if not os.path.exists("data/hlip_v_cur.dat"):
            return
        v_com = dict_array["hlip_v_cur"][:, 0]
    v_cmd = dict_array["com_plate_vx_cmd"][:, 0]
    if os.path.exists("data/com_plate_vx_err.dat"):
        v_err = dict_array["com_plate_vx_err"][:, 0]
    else:
        v_err = v_com - v_cmd
    valid = np.isfinite(t) & np.isfinite(v_err) & (t > 0.0)
    rmse = float(np.sqrt(np.mean(v_err[valid] * v_err[valid]))) if np.any(valid) else float("nan")
    v_cmd_nom = float(np.nanmedian(v_cmd[valid])) if np.any(valid) else float("nan")
    print(
        "Controller input velocity RMSE along plate vs commanded nominal "
        f"({v_cmd_nom:.3f} m/s): {rmse:.6f} m/s"
    )

    plt.figure("Velocity error from nominal", figsize=(9, 6))
    ax1 = plt.subplot(2, 1, 1)
    ax1.plot(t, v_cmd, "--", label="commanded nominal")
    ax1.plot(t, v_com, "-", label="controller input velocity along plate")
    ax1.set_ylabel("m/s")
    ax1.set_title("Controller input velocity along physical plate x-axis")
    ax1.legend(loc="best", fontsize=8)
    ax1.grid(True, "both", "both")

    ax2 = plt.subplot(2, 1, 2)
    ax2.plot(t, v_err, "-", color="C3", label=f"error, RMSE = {rmse:.4f} m/s")
    ax2.axhline(0.0, color="k", linewidth=0.8, linestyle="--")
    ax2.set_xlabel("t (sec)")
    ax2.set_ylabel("actual - commanded (m/s)")
    ax2.legend(loc="best", fontsize=8)
    ax2.grid(True, "both", "both")
    plt.tight_layout()
    plt.savefig("data/velocity_error_from_nominal.png", dpi=600)


def plot_vx_sc_minus_vx_plate():
    r"""ROM sagittal x velocity relative to plate: $v_{sc}^x - v_{plate}$.

    $v_{sc}^x$ is the world x component of (CoM − stance foot) linear velocity from
    Pinocchio — the same quantity as ``v_com_minus_stance`` in the log (``get_ROM_state``).
    $v_{plate}$ is ``plate_world_vx`` (slide joint $\dot{q}$ along world x).
    """
    req = ("t", "v_com_minus_stance", "plate_world_vx")
    if not all(k in dict_array for k in req):
        return

    t = dict_array["t"][:, 0]
    vx_sc = dict_array["v_com_minus_stance"][:, 0]
    vx_plate = dict_array["plate_world_vx"][:, 0]
    diff = vx_sc - vx_plate

    plt.figure(r"$v_{sc}^x - v_{plate}$ (ROM x rel. plate)", figsize=(9, 4))
    ax = plt.gca()
    ax.plot(t, diff, "-", color="C0", label=r"$v_{sc}^x - v_{plate}$")
    ax.axhline(0.0, color="k", linewidth=0.8, linestyle="--")
    ax.set_xlabel("t (s)")
    ax.set_ylabel("m/s")
    ax.set_title(
        r"Sagittal ROM velocity minus plate: $v_{sc}^x - v_{plate}$ "
        r"(same $v_{sc}^x$ as logged $v_{\mathrm{com}}^x - v_{\mathrm{stance}}^x$)"
    )
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, "both", "both")
    plt.tight_layout()
    plt.savefig("data/vx_sc_minus_vx_plate.png", dpi=600)


def plot_task_space_error():
    """Task space error: desired - actual for base orientation and arms (no COM z_sc, no swing foot)."""
    if "hd" not in dict_array or "h_cur" not in dict_array:
        return
    t = dict_array["t"][:, 0]
    err = dict_array["hd"] - dict_array["h_cur"]
    fig, axes = plt.subplots(3, 4, figsize=(16, 8))
    fig.suptitle(
        "Task space error — base & arms only (excl. z_sc, swing foot)",
        fontsize=12,
    )
    axes_flat = axes.flatten()
    for k, i in enumerate(_TASK_SPACE_BASE_ARM_INDICES):
        ax = axes_flat[k]
        ax.plot(t, err[:, i], "-")
        ax.set_title(_TASK_SPACE_BASE_ARM_LABELS[k], fontsize=9)
        ax.grid(True, "both", "both")
    axes_flat[-1].set_visible(False)
    plt.tight_layout()


def plot_task_space_tracking_base_arms():
    """Desired vs actual for base orientation and arm task space (excl. z_sc and swing foot)."""
    _plot_task_space_tracking_base_arms_only(
        None,
        "Task space tracking — base & arms (des vs act)",
    )


def plot_swing_foot_tracking_error():
    """Swing foot tracking error: desired - actual (position in m, orientation in deg)."""
    if "hd" not in dict_array or "h_cur" not in dict_array:
        return
    t = dict_array["t"][:, 0]
    err = dict_array["hd"] - dict_array["h_cur"]
    # Swing foot: indices 4-9 = x_sf, y_sf, z_sf, swf roll, swf pitch, swf yaw
    fig, axes = plt.subplots(2, 3, figsize=(12, 6))
    fig.suptitle("Swing foot tracking error (desired - actual)", fontsize=12)
    # Position (m)
    labels_pos = ["x_{sf} (m)", "y_{sf} (m)", "z_{sf} (m)"]
    for j, (idx, label) in enumerate(zip([4, 5, 6], labels_pos)):
        ax = axes[0, j]
        ax.plot(t, err[:, idx], '-')
        ax.set_title(label)
        ax.set_ylabel("des - actual")
        ax.set_ylim(-0.1, 0.1)
        ax.grid(True, 'both', 'both')
    # Orientation (deg)
    labels_ori = ["swf roll (deg)", "swf pitch (deg)", "swf yaw (deg)"]
    for j, (idx, label) in enumerate(zip([7, 8, 9], labels_ori)):
        ax = axes[1, j]
        ax.plot(t, err[:, idx] * 180 / np.pi, '-')
        ax.set_title(label)
        ax.set_ylabel("des - actual")
        ax.grid(True, 'both', 'both')
    plt.tight_layout()


def plot_swing_foot_tracking():
    """Swing foot: desired vs actual (position m, orientation deg)."""
    if "hd" not in dict_array or "h_cur" not in dict_array:
        return
    t = dict_array["t"][:, 0]
    hd = dict_array["hd"]
    hc = dict_array["h_cur"]
    fig, axes = plt.subplots(2, 3, figsize=(12, 6))
    fig.suptitle("Swing foot tracking (des vs act)", fontsize=12)
    labels_pos = ["x_{sf} (m)", "y_{sf} (m)", "z_{sf} (m)"]
    for j, idx in enumerate([4, 5, 6]):
        ax = axes[0, j]
        ax.plot(t, hd[:, idx], "--", label="des")
        ax.plot(t, hc[:, idx], "-", label="act")
        ax.set_title(labels_pos[j])
        ax.set_ylim(-0.1, 0.1)
        ax.grid(True, "both", "both")
    labels_ori = ["swf roll (deg)", "swf pitch (deg)", "swf yaw (deg)"]
    for j, idx in enumerate([7, 8, 9]):
        ax = axes[1, j]
        ax.plot(t, hd[:, idx] * 180 / np.pi, "--", label="des")
        ax.plot(t, hc[:, idx] * 180 / np.pi, "-", label="act")
        ax.set_title(labels_ori[j])
        ax.grid(True, "both", "both")
    axes[0, 0].legend(loc="upper right", fontsize=8)
    plt.tight_layout()


def plot_hlip_forward_touchdown():
    """HLIP commanded forward touchdown: stance-frame swing-foot x placement (u_x) vs realized u_x_act."""
    if "u_x" not in dict_array:
        return
    plt.figure("HLIP forward touchdown", figsize=(8, 4))
    t = dict_array["t"][:, 0]
    plt.plot(t, dict_array["u_x"][:, 0], "--", label="u_x des (HLIP / step cmd)")
    plt.plot(t, dict_array["u_x_act"][:, 0], "-", label="u_x act (touchdown)")
    plt.xlabel("t (sec)")
    plt.ylabel("forward placement w.r.t. stance foot (m)")
    plt.legend()
    plt.grid(True, "both", "both")
    plt.tight_layout()


def plot_hlip_pre_impact_estimate():
    """Biased HLIP: H-LIP–propagated pre-impact sagittal CoM state (x_pre_est, v_pre_est) fed to deadbeat law."""
    if "x_pre_est" not in dict_array or "v_pre_est" not in dict_array:
        return
    plt.figure("HLIP pre-impact estimate", figsize=(8, 5))
    t = dict_array["t"][:, 0]
    ax1 = plt.subplot(2, 1, 1)
    ax1.plot(t, dict_array["x_pre_est"][:, 0], "-", label="x_pre_est")
    ax1.set_ylabel("x_pre_est (m)")
    ax1.legend(loc="upper right", fontsize=8)
    ax1.grid(True, "both", "both")
    ax2 = plt.subplot(2, 1, 2)
    ax2.plot(t, dict_array["v_pre_est"][:, 0], "-", color="C1", label="v_pre_est")
    ax2.set_xlabel("t (sec)")
    ax2.set_ylabel("v_pre_est (m/s)")
    ax2.legend(loc="upper right", fontsize=8)
    ax2.grid(True, "both", "both")
    plt.tight_layout()


def plot_hlip_orbit_energy_pre_desired():
    """Biased HLIP orbital energy from latched pre-impact state vs desired z*."""
    req = ("t", "x_pre", "vx_pre", "hlip_z_star", "x_eq")
    if not all(k in dict_array for k in req):
        return

    _, z0, g = _model_params()
    t = dict_array["t"][:, 0]
    x_eq = dict_array["x_eq"][:, 0]

    if "plate_imu_acc" in dict_array and np.any(dict_array["plate_imu_acc"]):
        geff_z = np.maximum(np.abs(dict_array["plate_imu_acc"][:, 2]), 1.0)
    else:
        geff_z = np.full_like(t, float(g), dtype=float)
    alpha_sq = geff_z / float(z0)

    x_pre = dict_array["x_pre"][:, 0]
    vx_pre = dict_array["vx_pre"][:, 0]
    x_star = dict_array["hlip_z_star"][:, 0]
    v_star = dict_array["hlip_z_star"][:, 1]

    e_pre = 0.5 * (vx_pre * vx_pre - alpha_sq * (x_pre - x_eq) ** 2)
    e_pre_no_xeq = 0.5 * (vx_pre * vx_pre - alpha_sq * x_pre ** 2)
    e_pre_des = 0.5 * (v_star * v_star - alpha_sq * (x_star - x_eq) ** 2)

    plt.figure("HLIP orbit energy from pre desired", figsize=(9, 7))
    ax1 = plt.subplot(3, 1, 1)
    ax1.plot(t, e_pre, "-", label=r"$E_{pre}$ from latched $(x_{pre}, v_{x,pre})$")
    ax1.plot(t, e_pre_no_xeq, "-.", label=r"$E_{pre}$ no $x_{eq}$ shift")
    ax1.plot(t, e_pre_des, "--", label=r"$E^*_{pre}$ from $(x^*_{pre}, v^*_{pre})$")
    if "x_pre_est" in dict_array and "v_pre_est" in dict_array:
        x_pre_est = dict_array["x_pre_est"][:, 0]
        v_pre_est = dict_array["v_pre_est"][:, 0]
        e_pre_est = 0.5 * (v_pre_est * v_pre_est - alpha_sq * (x_pre_est - x_eq) ** 2)
        ax1.plot(t, e_pre_est, ":", color="C4", label=r"$E_{pre,est}$ from propagated estimate")
    ax1.set_ylabel(r"orbit energy (m$^2$/s$^2$)")
    ax1.legend(loc="upper right", fontsize=8)
    ax1.grid(True, "both", "both")

    ax2 = plt.subplot(3, 1, 2)
    ax2.plot(t, e_pre - e_pre_des, "-", color="C3", label=r"$E_{pre} - E^*_{pre}$")
    ax2.axhline(0.0, color="k", linewidth=0.8, alpha=0.5)
    ax2.set_ylabel("energy error")
    ax2.legend(loc="upper right", fontsize=8)
    ax2.grid(True, "both", "both")

    ax3 = plt.subplot(3, 1, 3)
    ax3.plot(t, x_star, "--", label=r"$x^*_{pre}$")
    ax3.plot(t, v_star, "--", color="C1", label=r"$v^*_{pre}$")
    ax3.plot(t, x_eq, ":", color="C2", label=r"$x_{eq}$")
    ax3.set_xlabel("t (sec)")
    ax3.set_ylabel("pre desired state")
    ax3.legend(loc="upper right", fontsize=8)
    ax3.grid(True, "both", "both")
    plt.tight_layout()
    plt.savefig("data/HLIP_orbit_energy_pre_desired.png", dpi=600)


def plot_hlip_x_cur_vs_x_actual():
    """HLIP x_cur vs x* = z*[0]; dotted purple: x_pre_est (propagate HLIP)."""
    if "hlip_x_cur" not in dict_array or "hlip_z_star" not in dict_array:
        return
    plt.figure("HLIP x_cur vs x* (z_star)", figsize=(8, 7))
    t = dict_array["t"][:, 0]
    x_star = dict_array["hlip_z_star"][:, 0]
    ax1 = plt.subplot(3, 1, 1)
    ax1.plot(t, dict_array["hlip_x_cur"][:, 0], "-", label="x_cur (into propagate HLIP)")
    ax1.plot(t, x_star, "--", label=r"$x^*$ from $z^*$ (x_des_pre)")
    if "x_pre_est" in dict_array:
        ax1.plot(
            t,
            dict_array["x_pre_est"][:, 0],
            ":",
            color="C4",
            linewidth=1.2,
            label="x_pre_est (propagate HLIP)",
        )
    ax1.set_ylabel("x (m)")
    ax1.legend(loc="upper right", fontsize=8)
    ax1.grid(True, "both", "both")
    ax1.set_title("Sagittal CoM relative to stance foot")
    ax2 = plt.subplot(3, 1, 2)
    diff = dict_array["hlip_x_cur"][:, 0] - x_star
    ax2.plot(t, diff, "-", color="C3", label=r"hlip_x_cur − $x^*$")
    ax2.set_ylabel("difference (m)")
    ax2.legend(loc="upper right", fontsize=8)
    ax2.grid(True, "both", "both")
    ax3 = plt.subplot(3, 1, 3)
    if "x_pre" in dict_array and "x_pre_est" in dict_array:
        err_pre = dict_array["x_pre"][:, 0] - dict_array["x_pre_est"][:, 0]
        ax3.plot(t, err_pre, "-", color="C5", label="x_pre − x_pre_est")
    ax3.set_xlabel("t (sec)")
    ax3.set_ylabel("error (m)")
    ax3.legend(loc="upper right", fontsize=8)
    ax3.grid(True, "both", "both")
    plt.tight_layout()


def plot_hlip_v_cur_vs_v_star():
    """HLIP v_cur vs v* = z*[1]; dotted purple: v_pre_est (propagate HLIP)."""
    if "hlip_v_cur" not in dict_array or "hlip_z_star" not in dict_array:
        return
    plt.figure("HLIP v_cur vs v* (z_star)", figsize=(8, 7))
    t = dict_array["t"][:, 0]
    v_star = dict_array["hlip_z_star"][:, 1]
    ax1 = plt.subplot(3, 1, 1)
    ax1.plot(t, dict_array["hlip_v_cur"][:, 0], "-", label="v_cur (into propagate HLIP)")
    ax1.plot(t, v_star, "--", label=r"$v^*$ from $z^*$ (v_des_pre)")
    if "v_pre_est" in dict_array:
        ax1.plot(
            t,
            dict_array["v_pre_est"][:, 0],
            ":",
            color="C4",
            linewidth=1.2,
            label="v_pre_est (propagate HLIP)",
        )
    ax1.set_ylabel("v (m/s)")
    ax1.legend(loc="upper right", fontsize=8)
    ax1.grid(True, "both", "both")
    ax1.set_title("Sagittal CoM velocity relative to stance foot")
    ax2 = plt.subplot(3, 1, 2)
    diff = dict_array["hlip_v_cur"][:, 0] - v_star
    ax2.plot(t, diff, "-", color="C3", label=r"hlip_v_cur − $v^*$")
    ax2.set_ylabel("difference (m/s)")
    ax2.legend(loc="upper right", fontsize=8)
    ax2.grid(True, "both", "both")
    ax3 = plt.subplot(3, 1, 3)
    if "vx_pre" in dict_array and "v_pre_est" in dict_array:
        err_pre = dict_array["vx_pre"][:, 0] - dict_array["v_pre_est"][:, 0]
        ax3.plot(t, err_pre, "-", color="C5", label="vx_pre − v_pre_est")
    ax3.set_xlabel("t (sec)")
    ax3.set_ylabel("error (m/s)")
    ax3.legend(loc="upper right", fontsize=8)
    ax3.grid(True, "both", "both")
    plt.tight_layout()


def plot_hlip_L_cur_vs_L_star():
    """ALIP/HLIP angular momentum L_cur vs L*; mirrors plot_hlip_v_cur_vs_v_star()."""
    if "hlip_v_cur" not in dict_array or "hlip_z_star" not in dict_array:
        return

    mass, z0, _ = _model_params()
    mz0 = float(mass) * float(z0)
    t = dict_array["t"][:, 0]

    if "alip_L_cur" in dict_array and np.any(dict_array["alip_L_cur"]):
        L_cur = dict_array["alip_L_cur"][:, 0]
    else:
        L_cur = mz0 * dict_array["hlip_v_cur"][:, 0]
    try:
        from digit_controller import BIASED_STEP_MODEL
        biased_step_model = str(BIASED_STEP_MODEL).upper()
    except Exception:
        biased_step_model = "HLIP"

    if biased_step_model == "ALIP":
        L_star = dict_array["hlip_z_star"][:, 1]
    else:
        L_star = mz0 * dict_array["hlip_z_star"][:, 1]

    plt.figure("HLIP/ALIP L_cur vs L* (z_star)", figsize=(8, 7))
    ax1 = plt.subplot(3, 1, 1)
    ax1.plot(t, L_cur, "-", label=r"$L_{cur}=m z_0 v_{cur}$")
    ax1.plot(t, L_star, "--", label=r"$L^*$ from $z^*$")
    if "alip_L_pre_est" in dict_array and np.any(dict_array["alip_L_pre_est"]):
        ax1.plot(
            t,
            dict_array["alip_L_pre_est"][:, 0],
            ":",
            color="C4",
            linewidth=1.2,
            label=r"$L_{pre,est}$",
        )
    elif "v_pre_est" in dict_array:
        ax1.plot(
            t,
            mz0 * dict_array["v_pre_est"][:, 0],
            ":",
            color="C4",
            linewidth=1.2,
            label=r"$L_{pre,est}=m z_0 v_{pre,est}$",
        )
    ax1.set_ylabel("L (kg m^2/s)")
    ax1.legend(loc="upper right", fontsize=8)
    ax1.grid(True, "both", "both")
    ax1.set_title("Sagittal angular momentum about stance foot")

    ax2 = plt.subplot(3, 1, 2)
    diff = L_cur - L_star
    ax2.plot(t, diff, "-", color="C3", label=r"$L_{cur} - L^*$")
    ax2.set_ylabel("difference")
    ax2.legend(loc="upper right", fontsize=8)
    ax2.grid(True, "both", "both")

    ax3 = plt.subplot(3, 1, 3)
    if "vx_pre" in dict_array and "v_pre_est" in dict_array:
        L_pre_ref = mz0 * dict_array["vx_pre"][:, 0]
        if "alip_L_pre_est" in dict_array and np.any(dict_array["alip_L_pre_est"]):
            L_pre_est = dict_array["alip_L_pre_est"][:, 0]
        else:
            L_pre_est = mz0 * dict_array["v_pre_est"][:, 0]
        err_pre = L_pre_ref - L_pre_est
        ax3.plot(t, err_pre, "-", color="C5", label=r"$m z_0 v_{pre} - L_{pre,est}$")
    ax3.set_xlabel("t (sec)")
    ax3.set_ylabel("error")
    ax3.legend(loc="upper right", fontsize=8)
    ax3.grid(True, "both", "both")
    plt.tight_layout()


def plot_hlip_K_z_star():
    """Biased HLIP deadbeat row K_db and reference pre-impact fixed point z* = [x*_pre, v*_pre]."""
    if "hlip_K" not in dict_array or "hlip_z_star" not in dict_array:
        return
    plt.figure("HLIP K and z_star", figsize=(8, 5))
    t = dict_array["t"][:, 0]
    K = dict_array["hlip_K"]
    zs = dict_array["hlip_z_star"]
    ax1 = plt.subplot(2, 1, 1)
    ax1.plot(t, K[:, 0], "-", label="K[0] (=1)")
    ax1.plot(t, K[:, 1], "-", color="C1", label="K[1] (Td + coth-like term)")
    ax1.set_ylabel("K_db components")
    ax1.legend(loc="upper right", fontsize=8)
    ax1.grid(True, "both", "both")
    ax2 = plt.subplot(2, 1, 2)
    ax2.plot(t, zs[:, 0], "-", label="z*_x (x_des_pre)")
    ax2.plot(t, zs[:, 1], "-", color="C1", label="z*_v (v_des_pre)")
    ax2.set_xlabel("t (sec)")
    ax2.set_ylabel("z* (m, m/s)")
    ax2.legend(loc="upper right", fontsize=8)
    ax2.grid(True, "both", "both")
    plt.tight_layout()


#############################
# plot joint space tracking #
#############################

def plot_left_leg_joint_tracking():
    # left leg
    plt.figure("left leg - joint tracking")
    
    ezplot_joint(3,4, 1,"qjnt_d", 0,"qpin", 7,"left hip roll (deg)")
    ezplot_joint(3,4, 2,"qjnt_d", 1,"qpin", 8,"left hip yaw (deg)")
    ezplot_joint(3,4, 3,"qjnt_d", 2,"qpin", 9,"left hip pitch (deg)")
    ezplot_joint(3,4, 4,"qjnt_d", 3,"qpin",10,"left knee (deg)")
    
    ezplot_joint(3,4, 5,"qjnt_d", 4,"qpin",11,"left shin (deg)")
    ezplot_joint(3,4, 6,"qjnt_d", 5,"qpin",12,"left tarsus (deg)")
    ezplot_joint(3,4, 7,"qjnt_d", 6,"qpin",13,"left heel spring (deg)")
    ezplot_joint(3,4, 8,"qjnt_d", 7,"qpin",14,"left toe A (deg)")
    
    ezplot_joint(3,4, 9,"qjnt_d", 8,"qpin",15,"left toe B (deg)")
    ezplot_joint(3,4,10,"qjnt_d", 9,"qpin",16,"left toe pitch (deg)")
    ezplot_joint(3,4,11,"qjnt_d",10,"qpin",17,"left toe roll (deg)")
    
def plot_left_arm_joint_tracking():
    # left arm
    plt.figure("left arm - joint tracking")

    plt.subplot(2,4,1)
    plt.plot(dict_array["qjnt_d"][:,11]*180/np.pi,'--')
    plt.plot(dict_array["qpin"][:,18]*180/np.pi,'-')
    plt.title("left-shouler-roll (deg)")
    plt.subplot(2,4,5)
    plt.plot(dict_array["uf"][:,6])
    plt.title("joint torque (Nm)")

    plt.subplot(2,4,2)
    plt.plot(dict_array["qjnt_d"][:,12]*180/np.pi,'--')
    plt.plot(dict_array["qpin"][:,19]*180/np.pi,'-')
    plt.title("left-shouler-pitch (deg)")
    plt.subplot(2,4,6)
    plt.plot(dict_array["uf"][:,7])
    plt.title("joint torque (Nm)")

    plt.subplot(2,4,3)
    plt.plot(dict_array["qjnt_d"][:,13]*180/np.pi,'--')
    plt.plot(dict_array["qpin"][:,20]*180/np.pi,'-')
    plt.title("left-shouler-yaw (deg)")
    plt.subplot(2,4,7)
    plt.plot(dict_array["uf"][:,8])
    plt.title("joint torque (Nm)")

    plt.subplot(2,4,4)
    plt.plot(dict_array["qjnt_d"][:,14]*180/np.pi,'--')
    plt.plot(dict_array["qpin"][:,21]*180/np.pi,'-')
    plt.title("left-elbow (deg)")
    plt.subplot(2,4,8)
    plt.plot(dict_array["uf"][:,9])
    plt.title("joint torque (Nm)")

def plot_right_leg_joint_tracking():
    # right leg
    plt.figure("right leg - joint tracking")
    
    ezplot_joint(3,4, 1,"qjnt_d",15,"qpin",22,"right hip roll (deg)")
    ezplot_joint(3,4, 2,"qjnt_d",16,"qpin",23,"right hip yaw (deg)")
    ezplot_joint(3,4, 3,"qjnt_d",17,"qpin",24,"right hip pitch (deg)")
    ezplot_joint(3,4, 4,"qjnt_d",18,"qpin",25,"right knee (deg)")
    
    ezplot_joint(3,4, 5,"qjnt_d",19,"qpin",26,"right shin (deg)")
    ezplot_joint(3,4, 6,"qjnt_d",20,"qpin",27,"right tarsus (deg)")
    ezplot_joint(3,4, 7,"qjnt_d",21,"qpin",28,"right heel spring (deg)")
    ezplot_joint(3,4, 8,"qjnt_d",22,"qpin",29,"right toe A (deg)")
    
    ezplot_joint(3,4, 9,"qjnt_d",23,"qpin",30,"right toe B (deg)")
    ezplot_joint(3,4,10,"qjnt_d",24,"qpin",31,"right toe pitch (deg)")
    ezplot_joint(3,4,11,"qjnt_d",25,"qpin",32,"right toe roll (deg)")

def plot_right_arm_joint_tracking():
    # right arm
    plt.figure("right arm - joint tracking")

    plt.subplot(2,4,1)
    plt.plot(dict_array["qjnt_d"][:,26]*180/np.pi,'--')
    plt.plot(dict_array["qpin"][:,33]*180/np.pi,'-')
    plt.title("right-shouler-roll (deg)")
    plt.subplot(2,4,5)
    plt.plot(dict_array["uf"][:,16])
    plt.title("joint torque (Nm)")

    plt.subplot(2,4,2)
    plt.plot(dict_array["qjnt_d"][:,27]*180/np.pi,'--')
    plt.plot(dict_array["qpin"][:,34]*180/np.pi,'-')
    plt.title("right-shouler-pitch (deg)")
    plt.subplot(2,4,6)
    plt.plot(dict_array["uf"][:,17])
    plt.title("joint torque (Nm)")

    plt.subplot(2,4,3)
    plt.plot(dict_array["qjnt_d"][:,28]*180/np.pi,'--')
    plt.plot(dict_array["qpin"][:,35]*180/np.pi,'-')
    plt.title("right-shouler-yaw (deg)")
    plt.subplot(2,4,7)
    plt.plot(dict_array["uf"][:,18])
    plt.title("joint torque (Nm)")

    plt.subplot(2,4,4)
    plt.plot(dict_array["qjnt_d"][:,29]*180/np.pi,'--')
    plt.plot(dict_array["qpin"][:,36]*180/np.pi,'-')
    plt.title("right-elbow (deg)")
    plt.subplot(2,4,8)
    plt.plot(dict_array["uf"][:,19])
    plt.title("joint torque (Nm)")

# velocity tracking

def plot_left_leg_joint_vel_tracking():
    # left leg
    plt.figure("left leg - joint vel tracking")
    
    ezplot_joint(3,4, 1,"vpin_d", 6,"vpin", 6,"left hip roll (deg/s)")
    plt.ylim([-500, 500])
    ezplot_joint(3,4, 2,"vpin_d", 7,"vpin", 7,"left hip yaw (deg/s)")
    plt.ylim([-500, 500])
    ezplot_joint(3,4, 3,"vpin_d", 8,"vpin", 8,"left hip pitch (deg/s)")
    plt.ylim([-500, 500])
    ezplot_joint(3,4, 4,"vpin_d", 9,"vpin", 9,"left knee (deg/s)")
    plt.ylim([-500, 500])
    ezplot_joint(3,4, 5,"vpin_d",10,"vpin",10,"left shin (deg/s)")
    plt.ylim([-500, 500])
    ezplot_joint(3,4, 6,"vpin_d",11,"vpin",11,"left tarsus (deg/s)")
    plt.ylim([-500, 500])
    ezplot_joint(3,4, 7,"vpin_d",12,"vpin",12,"left heel spring (deg/s)")
    plt.ylim([-500, 500])
    ezplot_joint(3,4, 9,"vpin_d",13,"vpin",13,"left toe A (deg/s)")
    plt.ylim([-200, 200])
    ezplot_joint(3,4,10,"vpin_d",14,"vpin",14,"left toe B (deg/s)")
    plt.ylim([-200, 200])
    ezplot_joint(3,4,11,"vpin_d",15,"vpin",15,"left toe pitch (deg/s)")
    plt.ylim([-200, 200])
    ezplot_joint(3,4,12,"vpin_d",16,"vpin",16,"left toe roll (deg/s)")
    plt.ylim([-200, 200])

def plot_right_leg_joint_vel_tracking():
    # right leg
    plt.figure("right leg - joint vel tracking")
    
    ezplot_joint(3,4, 1,"vpin_d",21,"vpin",21,"right hip roll (deg/s)")
    plt.ylim([-500, 500])
    ezplot_joint(3,4, 2,"vpin_d",22,"vpin",22,"right hip yaw (deg/s)")
    plt.ylim([-500, 500])
    ezplot_joint(3,4, 3,"vpin_d",23,"vpin",23,"right hip pitch (deg/s)")
    plt.ylim([-500, 500])
    ezplot_joint(3,4, 4,"vpin_d",24,"vpin",24,"right knee (deg/s)")
    plt.ylim([-500, 500])
    ezplot_joint(3,4, 5,"vpin_d",25,"vpin",25,"right shin (deg/s)")
    plt.ylim([-500, 500])
    ezplot_joint(3,4, 6,"vpin_d",26,"vpin",26,"right tarsus (deg/s)")
    plt.ylim([-500, 500])
    ezplot_joint(3,4, 7,"vpin_d",27,"vpin",27,"right heel spring (deg/s)")
    plt.ylim([-500, 500])
    ezplot_joint(3,4, 9,"vpin_d",28,"vpin",28,"right toe A (deg/s)")
    plt.ylim([-200, 200])
    ezplot_joint(3,4,10,"vpin_d",29,"vpin",29,"right toe B (deg/s)")
    plt.ylim([-200, 200])
    ezplot_joint(3,4,11,"vpin_d",30,"vpin",30,"right toe pitch (deg/s)")
    plt.ylim([-200, 200])
    ezplot_joint(3,4,12,"vpin_d",31,"vpin",31,"right toe roll (deg/s)")
    plt.ylim([-200, 200])


# step length command 
def plot_step_length_cmd():
    plt.figure("step length command")
    
    plt.subplot(1,2,1)
    plt.plot(dict_array["t"][:,0],dict_array["u_x"][:,0],'--')
    plt.plot(dict_array["t"][:,0],dict_array["u_x_act"][:,0],'-')
    plt.ylabel("step length x (m)")
    plt.grid(True,'both','both')
    
    plt.subplot(1,2,2)
    plt.plot(dict_array["t"][:,0],dict_array["u_y"][:,0],'--')
    plt.plot(dict_array["t"][:,0],dict_array["u_y_act"][:,0],'-')
    plt.ylabel("step length y (m)")
    plt.grid(True,'both','both')

# plot averaged walking speed tracking!!
def plot_step_output_tracking():
    plt.figure("step output tracking")
    
    plt.subplot(2,2,1)
    plt.plot(dict_array["t"][:,0],dict_array["output_x_des"][:,0],'--')
    plt.plot(dict_array["t"][:,0],dict_array["output_x_act"][:,0],'-')
    plt.xlabel("t (sec)")
    plt.ylabel("traveled distance in x (m)")
    plt.grid(True,'both','both')
    
    plt.subplot(2,2,3)
    plt.plot(dict_array["t"][:,0],dict_array["output_x_des"][:,1],'--')
    plt.plot(dict_array["t"][:,0],dict_array["output_x_act"][:,1],'-')
    plt.xlabel("t (sec)")
    plt.ylabel("velocity jump in x (m/s)")
    plt.grid(True,'both','both')
    
    plt.subplot(2,2,2)
    plt.plot(dict_array["t"][:,0],dict_array["output_y_des"][:,0],'--')
    plt.plot(dict_array["t"][:,0],dict_array["output_y_act"][:,0],'-')
    plt.xlabel("t (sec)")
    plt.ylabel("traveled distance in y (m)")
    plt.grid(True,'both','both')
    
    plt.subplot(2,2,4)
    plt.plot(dict_array["t"][:,0],dict_array["output_y_des"][:,1],'--')
    plt.plot(dict_array["t"][:,0],dict_array["output_y_act"][:,1],'-')
    plt.xlabel("t (sec)")
    plt.ylabel("D_mean in y (m)")
    plt.grid(True,'both','both')

# GRF
def plot_GRF():
    plt.figure("GRF")
    
    plt.plot(dict_array["t"][:,0],dict_array["L_GRF"][:,0],"r-")
    plt.plot(dict_array["t"][:,0],dict_array["R_GRF"][:,0],"b-")
    plt.plot(dict_array["t"][:,0],dict_array["foot_state"][:,0]*300 - 300,"k-")
    plt.plot(dict_array["t"][:,0],dict_array["GRF_mlc"][:,0],"r--")
    # plt.plot(dict_array["GRF_mlc"][:,1])
    plt.plot(dict_array["t"][:,0],dict_array["GRF_mlc"][:,2],"b--")
    # plt.plot(dict_array["GRF_mlc"][:,3])
    plt.plot(dict_array["t"][:,0],dict_array["pre_impact"][:,0]*200,"g--")
    plt.plot(dict_array["t"][:,0],dict_array["double_support"][:,0]*100,"y--")
    plt.grid(True,'both','both')
    
    plt.xlabel("t (sec)")
    plt.ylabel("GRF (N)")
    plt.legend(["FzL_mj","FzR_mj","foot state","FzL_mlc","FzR_mlc","pre_impact","DS"])

# ankle torque
def plot_ankle_torque():
    plt.figure("Ankle torque",figsize=(8, 6))
    
    plt.plot(dict_array["t"][:,0],dict_array["tau_ax"][:,0],"r-")
    plt.plot(dict_array["t"][:,0],dict_array["tau_ay"][:,0],"b-")
    plt.plot(dict_array["t"][:,0],dict_array["foot_state"][:,0]*1,"k-")
    # plt.plot(dict_array["t"][:,0],dict_array["pre_impact"][:,0]*10,"g--")
    # plt.plot(dict_array["t"][:,0],dict_array["double_support"][:,0]*10,"y--")
    
    plt.xlabel("t (sec)")
    plt.ylabel("ankle torque (Nm)")
    plt.legend(["tau ax","tau ay","foot state"]) #,"pre_impact","DS"])
    plt.grid(True, 'both','both')

    plt.savefig("data/Ankle_torque.png", dpi=600)     # Save as PNG


def plot_ankle_torque_components():
    """Plot decomposition of tau_ay into P and D components.

    Only the actually simulated portion is shown (trim zeros at the tail),
    so the curves match the raw data in the *.dat files.
    """
    # Use nonzero time (or any nonzero torque) to detect the valid simulation horizon
    t_full = dict_array["t"][:, 0]
    tauP_full = dict_array["tau_y_P"][:, 0]
    tauD_full = dict_array["tau_y_D"][:, 0]
    tau_tot_full = dict_array["tau_ay"][:, 0]

    # Find last index where *any* of the signals is nonzero
    nonzero_mask = (np.abs(tauP_full) > 1e-6) | (np.abs(tauD_full) > 1e-6) | (np.abs(tau_tot_full) > 1e-6)
    if np.any(nonzero_mask):
        end_idx = np.argmax(nonzero_mask[::-1])
        end_idx = len(t_full) - end_idx
        t = t_full[:end_idx]
        tauP = tauP_full[:end_idx]
        tauD = tauD_full[:end_idx]
        tau_tot = tau_tot_full[:end_idx]
    else:
        # Fallback: nothing logged, use full arrays (will all be zero)
        t = t_full
        tauP = tauP_full
        tauD = tauD_full
        tau_tot = tau_tot_full

    plt.figure("Ankle torque components (tau_ay)", figsize=(8, 6))

    plt.plot(t, tauP, "b-", label="tau_y_P (P-term)")
    plt.plot(t, tauD, "r-", label="tau_y_D (D-term)")
    plt.plot(t, tau_tot, "k--", label="tau_ay (total)")

    plt.xlabel("t (sec)")
    plt.ylabel("ankle torque about y (Nm)")
    plt.legend()
    plt.ylim([-150, 150])
    plt.grid(True, "both", "both")

    plt.savefig("data/Ankle_torque_components.png", dpi=600)


def plot_ankle_torque_all_components():
    """Plot all tau_ay components: gravity, bias, feedforward, P, D, and total."""
    t_full = dict_array["t"][:, 0]
    grav  = dict_array["tau_y_gravity"][:, 0]
    bias  = dict_array["tau_y_bias"][:, 0]
    ff    = dict_array["tau_y_ff"][:, 0]
    P     = dict_array["tau_y_P"][:, 0]
    D     = dict_array["tau_y_D"][:, 0]
    total = dict_array["tau_ay"][:, 0]

    nonzero_mask = (np.abs(total) > 1e-6) | (np.abs(grav) > 1e-6)
    if np.any(nonzero_mask):
        end_idx = len(t_full) - np.argmax(nonzero_mask[::-1])
        t     = t_full[:end_idx]
        grav  = grav[:end_idx]
        bias  = bias[:end_idx]
        ff    = ff[:end_idx]
        P     = P[:end_idx]
        D     = D[:end_idx]
        total = total[:end_idx]
    else:
        t = t_full

    plt.figure("Ankle torque all components (tau_ay)", figsize=(10, 6))
    plt.plot(t, grav,  label="gravity")
    plt.plot(t, bias,  label="bias (geff_x)")
    plt.plot(t, ff,    label="feedforward")
    plt.plot(t, P,     label="P-term")
    plt.plot(t, D,     label="D-term")
    plt.plot(t, total, "k--", linewidth=1.5, label="tau_ay (total)")

    plt.xlabel("t (sec)")
    plt.ylabel("ankle torque about y (Nm)")
    plt.legend()
    plt.grid(True, "both", "both")
    plt.savefig("data/Ankle_torque_all_components.png", dpi=600)


def plot_ankle_torque_non_pd_sum():
    """Gravity + bias + feedforward along tau_y (sagittal ankle), excluding PD (and not adaptive)."""
    req = ("t", "tau_y_gravity", "tau_y_bias", "tau_y_ff", "tau_y_PD", "tau_ay")
    if not all(k in dict_array for k in req):
        return
    t_full = dict_array["t"][:, 0]
    grav = dict_array["tau_y_gravity"][:, 0]
    bias = dict_array["tau_y_bias"][:, 0]
    ff = dict_array["tau_y_ff"][:, 0]
    pd = dict_array["tau_y_PD"][:, 0]
    total = dict_array["tau_ay"][:, 0]
    non_pd = grav + bias + ff

    nonzero_mask = (np.abs(total) > 1e-6) | (np.abs(non_pd) > 1e-6)
    if np.any(nonzero_mask):
        end_idx = len(t_full) - np.argmax(nonzero_mask[::-1])
        t = t_full[:end_idx]
        grav = grav[:end_idx]
        bias = bias[:end_idx]
        ff = ff[:end_idx]
        pd = pd[:end_idx]
        non_pd = non_pd[:end_idx]
        total = total[:end_idx]
    else:
        t = t_full

    plt.figure(r"Ankle $\tau_y$: non-PD sum (grav + bias + FF)", figsize=(10, 6))
    plt.plot(t, non_pd, "g-", linewidth=2.0, label=r"$\tau_{grav}+\tau_{bias}+\tau_{FF}$")
    plt.plot(t, grav, alpha=0.5, label="gravity")
    plt.plot(t, bias, alpha=0.5, label="bias")
    plt.plot(t, ff, alpha=0.5, label="feedforward")
    plt.plot(t, pd, "m--", linewidth=1.0, label=r"$\tau_{PD}$ (excluded from sum)")
    plt.plot(t, total, "k--", linewidth=1.2, label=r"$\tau_{ay}$ total")

    plt.xlabel("t (sec)")
    plt.ylabel("ankle torque about y (Nm)")
    plt.legend(loc="upper right", fontsize=8)
    plt.grid(True, "both", "both")
    plt.tight_layout()
    plt.savefig("data/Ankle_torque_non_PD_sum.png", dpi=600)


# Residuals: r_unbiased = m*z0*ẍ - (m*g*x + τ_y),  r_biased = m*z0*ẍ - (m*g_eff,z(t)*x - m*z0*g_eff,x(t) + τ_y)
# ẍ, x along projected sagittal (proj_plate_* / hlip_*). g_eff(t) matches digit_controller's plate-IMU conversion.
def _model_params():
    from HighLevelCtrler.AccCtrler import AccHighController
    _c = AccHighController()
    return _c.mass, _c.z_sc_d, _c.g


def plot_biased_residual():
    """
    Plot unbiased and biased residuals with time-varying g_eff from plate IMU (as set in sim).
    r_unbiased(t) = m*z0*ẍ - (m*g*x + τ_y)
    r_biased(t)   = m*z0*ẍ - (m*g_eff,z(t)*x - m*z0*g_eff,x(t) + τ_y)
    x, ẍ consistent with biased-HLIP sagittal: projected ROM (proj_plate_* preferred; else hlip_*; else fallback output_x_act).
    g_eff(t) from plate IMU uses measured pitch for the gravity-tilt part and
    flips only the translational acceleration part, matching digit_controller.
    """
    mass, z0, g = _model_params()
    t = dict_array["t"][:, 0]
    if "proj_plate_x" in dict_array and "proj_plate_vx" in dict_array:
        x = dict_array["proj_plate_x"][:, 0]
        vx = dict_array["proj_plate_vx"][:, 0]
    elif "hlip_x_cur" in dict_array and "hlip_v_cur" in dict_array:
        x = dict_array["hlip_x_cur"][:, 0]
        vx = dict_array["hlip_v_cur"][:, 0]
    else:
        x = dict_array["output_x_act"][:, 0]
        vx = dict_array["output_x_act"][:, 1]
    tau_y = dict_array["tau_ay"][:, 0]
    x_ddot = np.gradient(vx, t)

    # Time-varying g_eff from plate IMU (matches update_biased_hlip_from_ground_imu)
    plate_acc = dict_array["plate_imu_acc"]
    plate_pitch = dict_array["drs_act"][:, 4] if "drs_act" in dict_array else np.zeros_like(t)
    tilt_geff_x = -float(g) * np.sin(plate_pitch)
    geff_x = 2.0 * tilt_geff_x - plate_acc[:, 0]
    geff_z = np.maximum(np.abs(plate_acc[:, 2]), 1.0)

    r_unbiased = mass * z0 * x_ddot - (mass * g * x + tau_y)
    r_biased = mass * z0 * x_ddot - (mass * geff_z * x - mass * z0 * geff_x + tau_y)

    plt.figure("Residual (unbiased vs biased)", figsize=(8, 5))
    plt.plot(t, r_unbiased, "b-", label=r"$r_{unbiased}$")
    plt.plot(t, r_biased, "r-", label=r"$r_{biased}$")
    plt.xlabel("t (sec)")
    plt.ylabel("residual (Nm)")
    plt.ylim(-100, 100)
    plt.legend()
    plt.grid(True, "both", "both")
    plt.tight_layout()
    plt.savefig("data/Residual_biased_constant_geff.png", dpi=600)


def plot_ground_imu_output():
    """Compare logged geff channels."""
    req = ("geff_plate", "geff_torso_projected")
    missing = [name for name in req if name not in dict_array]
    if missing:
        print(f"plot_ground_imu_output skipped; missing {missing} (re-run test_mujoco_pbc.py)")
        return

    t = dict_array["t"][:, 0]
    geff_plate = dict_array["geff_plate"]
    geff_torso = dict_array["geff_torso_projected"]

    plt.figure("ground IMU geff compare", figsize=(9, 6))
    plt.subplot(2, 1, 1)
    plt.plot(t, geff_plate[:, 0], "-", label=r"$g_{\mathrm{eff},x}$ plate (logged)")
    plt.plot(t, geff_torso[:, 0], "--", label=r"$g_{\mathrm{eff},x}$ torso_projection_estimation")
    plt.ylabel(r"$g_{\mathrm{eff},x}$ (m/s$^2$)")
    plt.legend(fontsize=8)
    plt.grid(True, "both", "both")

    plt.subplot(2, 1, 2)
    plt.plot(t, geff_plate[:, 1], "-", label=r"$g_{\mathrm{eff},z}$ plate (logged)")
    plt.plot(t, geff_torso[:, 1], "--", label=r"$g_{\mathrm{eff},z}$ torso_projection_estimation")
    plt.xlabel("t (sec)")
    plt.ylabel(r"$g_{\mathrm{eff},z}$ (m/s$^2$)")
    plt.legend(fontsize=8)
    plt.grid(True, "both", "both")
    plt.tight_layout()
    plt.savefig("data/ground_imu_geff_compare.png", dpi=600)


def plot_imu_stance_acc_compare():
    """Compare plate IMU and torso IMU after both are rotated into stance frame."""
    req = ("t", "plate_imu_acc_stance", "torso_imu_acc_stance")
    missing = [name for name in req if name not in dict_array]
    if missing:
        print(f"plot_imu_stance_acc_compare skipped; missing {missing}")
        return

    t = dict_array["t"][:, 0]
    plate = dict_array["plate_imu_acc_stance"]
    torso = dict_array["torso_imu_acc_stance"]

    plt.figure("IMU specific force in stance frame", figsize=(9, 8))
    labels = ("x", "y", "z")
    for idx, axis in enumerate(labels):
        plt.subplot(3, 1, idx + 1)
        plt.plot(t, plate[:, idx], "-", label=f"plate IMU stance {axis}")
        plt.plot(t, torso[:, idx], "--", label=f"torso IMU stance {axis}")
        plt.ylabel(f"a_{axis} (m/s^2)")
        plt.legend()
        plt.grid(True, "both", "both")
    plt.xlabel("t (sec)")
    plt.tight_layout()
    plt.savefig("data/imu_stance_acc_compare.png", dpi=600)


def plot_torso_projection_components():
    """Compare torso-frame vectors before torso-projected geff projection."""
    req = (
        "t",
        "torso_proj_imu_acc",
        "torso_proj_fk_foot_over_torso_acc",
        "torso_proj_result_acc",
        "torso_proj_result_slope_acc",
    )
    missing = [name for name in req if name not in dict_array]
    if missing:
        print(f"plot_torso_projection_components skipped; missing {missing}")
        return

    t = dict_array["t"][:, 0]
    imu = dict_array["torso_proj_imu_acc"]
    fk = dict_array["torso_proj_fk_foot_over_torso_acc"]
    result = dict_array["torso_proj_result_acc"]

    plt.figure("Torso projection components", figsize=(10, 8))
    labels = ("x", "y", "z")
    for idx, axis in enumerate(labels):
        ax = plt.subplot(3, 1, idx + 1)
        ax.plot(t, imu[:, idx], "-", label=rf"$a_{{torso,{axis}}}$ torso IMU")
        ax.plot(t, fk[:, idx], "--", label=rf"$a_{{foot/torso,{axis}}}$ FK")
        ax.plot(t, result[:, idx], ":", label=rf"raw {axis}: torso IMU + FK")
        ax.set_ylabel(r"m/s$^2$")
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, "both", "both")
    plt.xlabel("t (sec)")
    plt.tight_layout()
    plt.savefig("data/torso_projection_components.png", dpi=600)


def plot_torso_fk_angles():
    """Plot FK/IMU sagittal pitch angles used by the torso/foot projection."""
    req = ("t", "torso_fk_angles")
    missing = [name for name in req if name not in dict_array]
    if missing:
        print(f"plot_torso_fk_angles skipped; missing {missing}")
        return

    t = dict_array["t"][:, 0]
    angles_deg = dict_array["torso_fk_angles"] * 180.0 / np.pi

    plt.figure("Torso/FK sagittal angles", figsize=(10, 5))
    plt.plot(t, angles_deg[:, 0], "-", label="torso/world pitch")
    plt.plot(t, angles_deg[:, 1], "--", linewidth=2.0, label="torso/foot pitch")
    plt.plot(t, angles_deg[:, 2], ":", linewidth=2.0, label="foot/world pitch (raw composition)")
    plt.plot(t, angles_deg[:, 3], "-.", linewidth=1.5, label="foot/world pitch (bias-corrected)")
    plt.xlabel("t (sec)")
    plt.ylabel("pitch (deg)")
    plt.legend(fontsize=8)
    plt.grid(True, "both", "both")
    plt.tight_layout()
    plt.savefig("data/torso_fk_angles.png", dpi=600)


def plot_geff_imu_compare():
    """Plot logged geff estimators.

    Plate is in physical plate-IMU frame. Orange is torso_projection_estimation.

    In torso-projected mode after warmup, orange is the geff selected by
    the controller update path.
    """
    req = ("t", "geff_plate", "geff_torso_projected")
    missing = [name for name in req if name not in dict_array]
    if missing:
        print(f"plot_geff_imu_compare skipped; missing {missing}")
        return

    t = dict_array["t"][:, 0]
    geff_plate = dict_array["geff_plate"]
    geff_torso = dict_array["geff_torso_projected"]

    valid = t > 1e-6
    if np.any(valid):
        n_valid = int(np.count_nonzero(valid))
        torso_ok = np.isfinite(geff_torso[valid, 0])
        if int(np.count_nonzero(torso_ok)) == 0:
            print("WARNING: geff_torso_projected has no finite samples. Re-run test_mujoco_pbc.py.")
        print(f"geff compare: {n_valid} logged samples, t in [{t[valid][0]:.2f}, {t[valid][-1]:.2f}] s")
        print(
            f"  torso_projection_estimation finite samples: {int(np.count_nonzero(torso_ok))}/{n_valid}"
        )

    plt.figure("g_eff compare", figsize=(9, 6))

    ax = plt.subplot(2, 1, 1)
    ax.plot(t, geff_plate[:, 0], "-", label=r"$g_{\mathrm{eff},x}$ plate IMU")
    ax.plot(
        t,
        geff_torso[:, 0],
        "--",
        label=r"$g_{\mathrm{eff},x}$ torso_projection_estimation",
    )
    ax.set_ylabel(r"$g_{\mathrm{eff},x}$ (m/s$^2$)")
    ax.legend(fontsize=8)
    ax.grid(True, "both", "both")

    ax = plt.subplot(2, 1, 2)
    ax.plot(t, geff_plate[:, 1], "-", label=r"$g_{\mathrm{eff},z}$ plate IMU")
    ax.plot(
        t,
        geff_torso[:, 1],
        "--",
        label=r"$g_{\mathrm{eff},z}$ torso_projection_estimation",
    )
    ax.set_xlabel("t (sec)")
    ax.set_ylabel(r"$g_{\mathrm{eff},z}$ (m/s$^2$)")
    ax.legend(fontsize=8)
    ax.grid(True, "both", "both")
    plt.tight_layout()
    plt.savefig("data/geff_imu_compare.png", dpi=600)


def plot_swing_foot_x_rel_stance():
    """Swing foot x position relative to stance foot (m).
    - Projected: from biased HLIP at every sim step (hd[:,4], desired trajectory).
    - Actual: current swing foot x in stance frame (h_cur[:,4]).
    """
    plt.figure("Swing foot x (relative to stance foot)")
    plt.plot(dict_array["t"][:,0], dict_array["hd"][:,4], "--", label="projected (biased HLIP)")
    plt.plot(dict_array["t"][:,0], dict_array["h_cur"][:,4], "-", label="actual")
    plt.xlabel("t (sec)")
    plt.ylabel("swing foot x w.r.t. stance foot (m)")
    plt.legend()
    plt.grid(True, 'both', 'both')

def ezplot3(a,b,c,name,ylabel):
    plt.subplot(a,b,c)
    plt.plot(dict_array["t"][:,0],dict_array[name][:,:])
    plt.xlabel('t [sec]')
    plt.ylabel(ylabel)
    plt.grid(True,'both','both')


# adaptive controller's plot
def plot_adaptive_controller_x_sc():
    plt.figure("x_sc adaptive plot")
    ezplot3(3,3,1,"ada_x_obsever_state",'observer state')
    ezplot3(3,3,2,"ada_x_compensator_state",'compensator state')
    ezplot3(3,3,3,"ada_x_regressor_state",'regressor state')
    ezplot3(3,3,4,"ada_x_compensator_coef",'compensator coef')
    ezplot3(3,3,5,"ada_x_covariance_matrix_norm",'covariance matrix norm')
    ezplot3(3,3,6,"ada_y_tau",'tau adaptive')
    
    plt.subplot(3,3,7)
    plt.plot(dict_array["t"][:,0],dict_array["x_sc_des"][:,0] - dict_array["x_sc_cur"][:,0])
    plt.xlabel('t [sec]')
    plt.ylabel('x_sc_err')
    plt.grid(True,'both','both')
    

def plot_adaptive_controller_y_sc():
    plt.figure("y_sc adaptive plot")
    ezplot3(3,3,1,"ada_y_obsever_state",'observer state')
    ezplot3(3,3,2,"ada_y_compensator_state",'compensator state')
    ezplot3(3,3,3,"ada_y_regressor_state",'regressor state')
    ezplot3(3,3,4,"ada_y_compensator_coef",'compensator coef')
    ezplot3(3,3,5,"ada_y_covariance_matrix_norm",'covariance matrix norm')
    ezplot3(3,3,6,"ada_x_tau",'tau adaptive')
    
    plt.subplot(3,3,7)
    plt.plot(dict_array["t"][:,0],dict_array["y_sc_des"][:,0] - dict_array["y_sc_cur"][:,0])
    plt.xlabel('t [sec]')
    plt.ylabel('y_sc_err')
    plt.grid(True,'both','both')
    
def plot_ankle_P_vs_FF_inputs():
    """What goes into P-term vs FF-term of ankle torque (sagittal).

    Top-left:   x_d (desired) vs x_sc (actual) -- PD uses this error
    Top-right:  x_eq over time
    Bottom-left: FF accel comparison: normal (alpha^2*x_d) vs biased (alpha^2*x_tilde)
    Bottom-right: tau_y_ff (the actual FF torque applied)
    """
    req = ("t", "x_pd", "x_ff", "x_ddot_ff", "x_ddot_biased", "x_sc_cur", "x_eq", "tau_y_ff")
    if not all(k in dict_array for k in req):
        return
    t = dict_array["t"][:, 0]
    x_pd_ref = dict_array["x_pd"][:, 0]
    x_d = dict_array["x_ff"][:, 0]
    x_ddot_ff = dict_array["x_ddot_ff"][:, 0]
    x_ddot_biased = dict_array["x_ddot_biased"][:, 0]
    x_cur = dict_array["x_sc_cur"][:, 0]
    x_eq = dict_array["x_eq"][:, 0]
    tau_ff = dict_array["tau_y_ff"][:, 0]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Ankle torque: P-input vs FF-input (sagittal)", fontsize=12)

    ax = axes[0, 0]
    ax.plot(t, x_pd_ref, "--", label=r"$x_{pd}=x_d+x_{eq}$ (PD ref)")
    ax.plot(t, x_d, ":", color="tab:cyan", label=r"$x_{ff}=x_d$ (FF pos ref)")
    ax.plot(t, x_cur, "-", label=r"$x_{sc}$ (actual)")
    ax.set_ylabel("m")
    ax.set_title("PD ref vs FF pos ref vs actual")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True, "both", "both")

    ax = axes[0, 1]
    ax.plot(t, x_eq, "-", color="tab:red")
    ax.set_ylabel("m")
    ax.set_title(r"$x_{eq}$ (equilibrium offset)")
    ax.grid(True, "both", "both")

    ax = axes[1, 0]
    ax.plot(t, x_ddot_ff, "-", label=r"$\alpha^2 x_d$ (normal, used)")
    ax.plot(t, x_ddot_biased, "--", label=r"$\alpha^2 x_{tilde}$ (biased, old)")
    ax.set_xlabel("t (sec)")
    ax.set_ylabel(r"m/s$^2$")
    ax.set_title("FF accel: normal vs biased")
    ax.legend(fontsize=8)
    ax.grid(True, "both", "both")

    ax = axes[1, 1]
    ax.plot(t, tau_ff, "-", color="tab:green")
    ax.set_xlabel("t (sec)")
    ax.set_ylabel("Nm")
    ax.set_title(r"$\tau_{y,ff}$ (applied FF torque)")
    ax.grid(True, "both", "both")

    plt.tight_layout()
    plt.savefig("data/Ankle_P_vs_FF_inputs.png", dpi=600)


def plot_rom_accel_meas_vs_des():
    r"""Sagittal: projected stance-relative COM accel vs biased-HLIP model at projected x."""
    req = ("t", "x_sc_ddot_meas", "x_sc_ddot_des_hlip")
    if not all(k in dict_array for k in req):
        return
    t = dict_array["t"][:, 0]
    # fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    _, ax = plt.subplots(1, 1, figsize=(10, 4))
    ax.plot(t, dict_array["x_sc_ddot_meas"][:, 0], "-", label=r"$\ddot{x}$ meas (proj.)")
    ax.plot(t, dict_array["x_sc_ddot_des_hlip"][:, 0], "--", label=r"$\ddot{x}$ des (H-LIP, proj. \(x\))")
    ax.set_xlabel("t (sec)")
    ax.set_ylabel(r"m/s$^2$")
    ax.set_title(r"Sagittal COM accel (projected): meas vs biased H-LIP \(\alpha^2(x-x_\mathrm{eq})\)")
    ax.legend(fontsize=9)
    ax.grid(True, "both", "both")
    plt.tight_layout()
    plt.savefig("data/ROM_accel_meas_vs_des.png", dpi=600)


def plot_xsc_ysc():
    plt.figure("ROM tracking",figsize=(8,6))
    
    plt.subplot(2,2,1) # x_sc
    plt.plot(dict_array["t"][:,0],dict_array["x_sc_des"][:,0],'--')
    plt.plot(dict_array["t"][:,0],dict_array["x_sc_cur"][:,0],'-')
    plt.title("x_{sc} (m)")
    plt.grid(True,'both','both')
    
    plt.subplot(2,2,3) # x_sc_error
    plt.plot(dict_array["t"][:,0],dict_array["x_sc_des"][:,0] - dict_array["x_sc_cur"][:,0],'-')
    plt.xlabel('t [sec]')
    plt.title("x_{sc} error (m)")
    plt.grid(True,'both','both')
    plt.ylim([-0.05, 0.05])
    
    plt.subplot(2,2,2) # y_sc
    plt.plot(dict_array["t"][:,0],dict_array["y_sc_des"][:,0],'--')
    plt.plot(dict_array["t"][:,0],dict_array["y_sc_cur"][:,0],'-')
    plt.title("y_{sc} (m)")
    plt.grid(True,'both','both')
    
    plt.subplot(2,2,4) # y_sc_error
    plt.plot(dict_array["t"][:,0],dict_array["y_sc_des"][:,0] - dict_array["y_sc_cur"][:,0],'-')
    plt.xlabel('t [sec]')
    plt.title("y_{sc} error (m)")
    plt.grid(True,'both','both')
    plt.ylim([-0.05, 0.05])
    
    plt.savefig("data/ROM_tracking.png", dpi=600)     # Save as PNG
    

def _logged_row_count():
    """Use data/t.dat size so plots match the latest sim even if SIM_LENGTH differs."""
    t_path = "data/t.dat"
    if os.path.exists(t_path):
        nbytes = os.path.getsize(t_path)
        if nbytes >= 4:
            return nbytes // 4
    return SIM_LENGTH


_LOG_N = _logged_row_count()

dict_array = {}
for name in DATA_DICT.keys():
    shape = (_LOG_N, DATA_DICT[name])
    filepath = "data/{}.dat".format(name)
    if os.path.exists(filepath):
        expected_bytes = int(np.prod(shape)) * 4
        if os.path.getsize(filepath) < expected_bytes:
            print(
                f"WARNING: {filepath} is smaller than expected for {_LOG_N} rows; "
                "re-run test_mujoco_pbc.py to refresh logs."
            )
            use_n = os.path.getsize(filepath) // (4 * DATA_DICT[name])
            shape = (use_n, DATA_DICT[name])
        arr = np.memmap(filepath, dtype="float32", mode="r", shape=shape)
        dict_array[name] = arr.copy()
    else:
        # Missing channel (new name or no sim yet): zeros + warning once.
        dict_array[name] = np.zeros(shape, dtype=np.float32)
        print(f"WARNING: missing log file {filepath} (filled with zeros)")

# Restrict plotting window: drop the first START_IDX sim steps so every plot
# only shows data logged from that step onward. All helpers index with
# dict_array["t"][:,0], so slicing every channel here keeps them consistent.
if START_IDX > 0:
    _start = min(START_IDX, SIM_LENGTH)
    for _name in list(dict_array.keys()):
        dict_array[_name] = dict_array[_name][_start:, :]


########
# plot #
########
#plot_task_space_tracking() #1
#plot_COM_error()
#plot_COM_tracking()
#plot_x_pd_ref_vs_x_sc()
#plot_COM_x_des_vs_actual_plus_xeq()
#plot_swing_foot_tracking_error()
#plot_swing_foot_tracking()
#plot_task_space_error()
#plot_hlip_forward_touchdown()
#plot_hlip_pre_impact_estimate()
#plot_hlip_K_z_star()
#plot_hlip_x_cur_vs_x_actual()
#plot_hlip_v_cur_vs_v_star()
#plot_hlip_L_cur_vs_L_star()
#plot_hlip_orbit_energy_pre_desired()
#plot_vx_sc_minus_vx_plate()
#plot_velocity_frame_diagnostics()
plot_velocity_error_from_nominal()
#plot_com_world_x_and_stance_world_x()

#plot_left_leg_joint_tracking()
#plot_left_leg_joint_vel_tracking()
#plot_left_leg_torque()
# # plot_left_leg_torque_compare()
# # plot_left_arm_joint_tracking()

# # plot_right_leg_joint_tracking()
# plot_right_leg_joint_vel_tracking()
#plot_right_leg_torque()
# # plot_right_leg_torque_compare()
# # plot_right_arm_joint_tracking()

# plot_GRF()

#plot_step_length_cmd()

# plot_step_output_tracking()

#plot_ankle_torque()
#plot_ankle_torque_components()
#plot_ankle_torque_all_components()
#plot_ankle_torque_non_pd_sum()
#plot_ankle_P_vs_FF_inputs()

#plot_ground_imu_output()
#plot_imu_stance_acc_compare()
#plot_geff_imu_compare()
#plot_torso_projection_components()
#plot_torso_fk_angles()

# plot_adaptive_controller_x_sc() #1
# plot_adaptive_controller_y_sc() #1

# plot_xsc_ysc()
#plot_rom_accel_meas_vs_des()

#plot_DRS_tracking() #1

# swing foot x relative to stance foot
# plot_swing_foot_x_rel_stance()

# biased vs unbiased residual with constant g_eff
# plot_biased_residual()

# # z_sf tracking
# # plt.figure("z_sf")
# # plt.plot(dict_array["t"][:,0],dict_array["hd"][:,6],'--')
# # plt.plot(dict_array["t"][:,0],dict_array["h_cur"][:,6])
# # plt.title("z_{sf} (m)")
# # plt.grid(True,'both','both')

########
# show #
########
plt.show()
