OPENQASM 2.0;
include "qelib1.inc";
gate ryy(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(-3.2330837) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
gate ryy_139854475612800(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(-0.21060996) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
gate ryy_139854475570768(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(-0.16366565) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
gate ryy_139854475569664(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(-0.50988424) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
gate ryy_139854475567936(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(0.26896375) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
gate ryy_139854475571152(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(-0.15978816) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
gate ryy_139854475625376(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(1.9867018) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
gate ryy_139854475625808(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(-0.37230501) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
gate ryy_139854475627824(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(-0.89551055) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
gate ryy_139854475624704(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(-0.22626139) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
gate ryy_139854475628256(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(-0.033207599) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
gate ryy_139854475627056(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(0.10777132) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
qreg q[4];
creg c[4];
h q[0];
h q[1];
h q[2];
h q[3];
ry(2.1296248) q[0];
ry(-1.5181471) q[1];
ry(0.13872829) q[2];
ry(-1.1797569) q[3];
ryy(-3.2330837) q[0],q[1];
ryy_139854475612800(-0.21060996) q[1],q[2];
ryy_139854475570768(-0.16366565) q[2],q[3];
h q[0];
h q[1];
h q[2];
h q[3];
ry(-0.52974117) q[0];
ry(0.96251577) q[1];
ry(0.27943829) q[2];
ry(-0.57181913) q[3];
ryy_139854475569664(-0.50988424) q[0],q[1];
ryy_139854475567936(0.26896375) q[1],q[2];
ryy_139854475571152(-0.15978816) q[2],q[3];
h q[0];
h q[1];
h q[2];
h q[3];
ry(-2.7936289) q[0];
ry(-0.71115452) q[1];
ry(0.52352196) q[2];
ry(-1.7105501) q[3];
ryy_139854475625376(1.9867018) q[0],q[1];
ryy_139854475625808(-0.37230501) q[1],q[2];
ryy_139854475627824(-0.89551055) q[2],q[3];
h q[0];
h q[1];
h q[2];
h q[3];
ry(0.83848536) q[0];
ry(-0.26984537) q[1];
ry(0.12306158) q[2];
ry(0.87575114) q[3];
ryy_139854475624704(-0.22626139) q[0],q[1];
ryy_139854475628256(-0.033207599) q[1],q[2];
ryy_139854475627056(0.10777132) q[2],q[3];
rx(-0.65426201) q[0];
rx(0.55532503) q[1];
rx(-0.22716226) q[2];
rx(3.1444271) q[3];
rz(-0.93972737) q[0];
rz(-1.4414353) q[1];
rz(-0.66043341) q[2];
rz(1.4802935) q[3];
crz(2.2663836) q[0],q[1];
crz(-2.1964617) q[1],q[2];
crz(-0.96564609) q[2],q[3];
rx(-1.1617337) q[0];
rx(0.14104094) q[1];
rx(-1.1400031) q[2];
rx(-1.0200173) q[3];
rz(-5.7589216e-07) q[0];
rz(3.5931067e-09) q[1];
rz(0.5074023) q[2];
rz(0.15682325) q[3];
crz(-0.2803981) q[0],q[1];
crz(-0.23289838) q[1],q[2];
crz(0.50838423) q[2],q[3];
measure q[0] -> c[0];
measure q[1] -> c[1];
measure q[2] -> c[2];
measure q[3] -> c[3];
