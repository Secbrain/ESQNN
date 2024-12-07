OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
creg c[4];
rx(-1.4078194) q[0];
rx(-0.080110811) q[1];
rx(0.51941246) q[2];
rx(1.1708889) q[3];
rx(2.1779797) q[0];
rx(1.7791979) q[1];
rx(0.25832492) q[2];
rx(-2.4340737) q[3];
rx(-0.34975004) q[0];
rx(-1.338056) q[1];
rx(-0.43891034) q[2];
rx(-0.58501744) q[3];
rx(1.8071492) q[0];
rx(-0.73262411) q[1];
rx(0.40939674) q[2];
rx(-0.58409548) q[3];
rx(-3.2662222) q[0];
rx(-0.18159026) q[1];
rx(-2.2485087) q[2];
rx(-1.1187389) q[3];
rz(-0.58965713) q[0];
rz(2.1773462) q[1];
rz(2.8494737) q[2];
rz(-2.1077893) q[3];
crz(0.43782613) q[0],q[3];
crz(-1.8248744) q[1],q[0];
crz(-1.5890068) q[2],q[1];
crz(-2.8456442) q[3],q[2];
rx(-1.8980319) q[0];
rx(-0.35588282) q[1];
rx(-1.4758266) q[2];
rx(-0.97871238) q[3];
rz(0.71608794) q[0];
rz(-2.281131) q[1];
rz(-1.185226) q[2];
rz(0.64499497) q[3];
crz(-0.53689319) q[0],q[3];
crz(-1.7373283) q[1],q[0];
crz(-1.9221121) q[2],q[1];
crz(1.5418187) q[3],q[2];
rx(-0.58327353) q[0];
rx(-0.16072309) q[1];
rx(-1.8302357) q[2];
rx(1.2843037) q[3];
rz(0.59169322) q[0];
rz(-0.49948102) q[1];
rz(-5.1369836e-05) q[2];
rz(-0.12949729) q[3];
crz(-3.8107567e-10) q[0],q[3];
crz(9.0657654e-10) q[1],q[0];
crz(-0.04938731) q[2],q[1];
crz(0.12530395) q[3],q[2];
measure q[0] -> c[0];
measure q[1] -> c[1];
measure q[2] -> c[2];
measure q[3] -> c[3];