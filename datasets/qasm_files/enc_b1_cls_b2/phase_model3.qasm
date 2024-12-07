OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
creg c[4];
rx(1.4451338) q[0];
rx(0.85641253) q[1];
rx(2.2180758) q[2];
rx(0.52316552) q[3];
rx(0.34664667) q[0];
rx(-0.19733144) q[1];
rx(-1.0545889) q[2];
rx(1.2779956) q[3];
rx(-0.17219013) q[0];
rx(0.52378845) q[1];
rx(0.05662182) q[2];
rx(0.42629614) q[3];
rx(0.57500505) q[0];
rx(-0.64172411) q[1];
rx(-2.2063985) q[2];
rx(-0.75080305) q[3];
rx(-3.0542014) q[0];
rx(-3.0456436) q[1];
rx(0.18983518) q[2];
rx(-1.5445483) q[3];
rz(-1.7269359) q[0];
rz(-4.6572528) q[1];
rz(-1.033522) q[2];
rz(0.57689691) q[3];
crz(2.0192335) q[0],q[1];
crz(-0.98555881) q[1],q[2];
crz(-3.6681287) q[2],q[3];
rx(-2.1578743) q[0];
rx(3.5143726) q[1];
rx(0.80208212) q[2];
rx(-0.36728236) q[3];
rz(-7.3037052e-05) q[0];
rz(1.465042e-09) q[1];
rz(-9.3599339e-10) q[2];
rz(-3.2229194e-10) q[3];
crz(-0.023060488) q[0],q[1];
crz(-7.3589229e-10) q[1],q[2];
crz(4.9143289e-10) q[2],q[3];
measure q[0] -> c[0];
measure q[1] -> c[1];
measure q[2] -> c[2];
measure q[3] -> c[3];