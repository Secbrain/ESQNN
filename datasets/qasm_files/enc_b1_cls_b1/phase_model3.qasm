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
rx(-0.2178548) q[0];
rx(-0.2020805) q[1];
rx(0.60369992) q[2];
rx(-1.3306358) q[3];
rz(-6.4143252e-10) q[0];
rz(-0.75145137) q[1];
rz(-0.13943295) q[2];
rz(0.72749835) q[3];
crz(0.09404546) q[0],q[1];
crz(2.2738229e-07) q[1],q[2];
crz(-0.060910422) q[2],q[3];
measure q[0] -> c[0];
measure q[1] -> c[1];
measure q[2] -> c[2];
measure q[3] -> c[3];
