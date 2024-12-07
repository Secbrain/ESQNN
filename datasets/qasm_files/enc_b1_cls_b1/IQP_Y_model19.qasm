OPENQASM 2.0;
include "qelib1.inc";
gate ryy(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(0.64304018) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
gate ryy_139907418937184(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(0.5565024) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
gate ryy_139907418935888(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(0.71402955) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
gate ryy_139907418939008(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(-0.12567776) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
gate ryy_139907418892272(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(0.55832815) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
gate ryy_139907418892224(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(1.3599416) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
gate ryy_139907351687952(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(-0.44803312) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
gate ryy_139907351687616(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(0.43328598) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
gate ryy_139907351690112(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(0.42445651) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
gate ryy_139907351651232(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(2.2800949) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
gate ryy_139907351651952(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(0.37105721) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
gate ryy_139907351652816(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(0.31398451) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
qreg q[4];
creg c[4];
h q[0];
h q[1];
h q[2];
h q[3];
ry(1.0132366) q[0];
ry(0.63463968) q[1];
ry(0.87687933) q[2];
ry(0.81428486) q[3];
ryy(0.64304018) q[0],q[1];
ryy_139907418937184(0.5565024) q[1],q[2];
ryy_139907418935888(0.71402955) q[2],q[3];
h q[0];
h q[1];
h q[2];
h q[3];
ry(0.19737062) q[0];
ry(-0.63676023) q[1];
ry(-0.87682635) q[2];
ry(-1.5509816) q[3];
ryy_139907418939008(-0.12567776) q[0],q[1];
ryy_139907418892272(0.55832815) q[1],q[2];
ryy_139907418892224(1.3599416) q[2],q[3];
h q[0];
h q[1];
h q[2];
h q[3];
ry(-0.78818357) q[0];
ry(0.56843752) q[1];
ry(0.76224029) q[2];
ry(0.55685395) q[3];
ryy_139907351687952(-0.44803312) q[0],q[1];
ryy_139907351687616(0.43328598) q[1],q[2];
ryy_139907351690112(0.42445651) q[2],q[3];
h q[0];
h q[1];
h q[2];
h q[3];
ry(1.2983627) q[0];
ry(1.7561308) q[1];
ry(0.21129246) q[2];
ry(1.4860185) q[3];
ryy_139907351651232(2.2800949) q[0],q[1];
ryy_139907351651952(0.37105721) q[1],q[2];
ryy_139907351652816(0.31398451) q[2],q[3];
rx(0.36787131) q[0];
rx(-1.0816613) q[1];
rx(-0.73629922) q[2];
rx(2.9048641) q[3];
rz(1.4956422) q[0];
rz(1.1102763) q[1];
rz(-2.280386) q[2];
rz(-3.7557614) q[3];
crx(-0.094084539) q[0],q[3];
crx(1.4269713) q[1],q[0];
crx(0.82486838) q[2],q[1];
crx(-2.0219193) q[3],q[2];
measure q[0] -> c[0];
measure q[1] -> c[1];
measure q[2] -> c[2];
measure q[3] -> c[3];
