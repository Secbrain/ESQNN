OPENQASM 2.0;
include "qelib1.inc";
gate ryy(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(-4.3579407) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
gate ryy_139854475744016(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(-2.140743) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
gate ryy_139854475743632(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(-0.61308616) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
gate ryy_139854475745168(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(1.0205148) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
gate ryy_139854475746752(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(-3.0007863) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
gate ryy_139854475745264(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(0.28826877) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
gate ryy_139854475222224(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(-1.3988656) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
gate ryy_139854475222848(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(2.639245) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
gate ryy_139854475219968(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(-0.12578748) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
gate ryy_139854492587680(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(0.15877706) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
gate ryy_139854492588304(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(0.28134722) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
gate ryy_139854492588880(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(0.90068722) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
qreg q[4];
creg c[4];
h q[0];
h q[1];
h q[2];
h q[3];
ry(-1.8736529) q[0];
ry(2.325906) q[1];
ry(-0.92039096) q[2];
ry(0.66611493) q[3];
ryy(-4.3579407) q[0],q[1];
ryy_139854475744016(-2.140743) q[1],q[2];
ryy_139854475743632(-0.61308616) q[2],q[3];
h q[0];
h q[1];
h q[2];
h q[3];
ry(-0.44026491) q[0];
ry(-2.3179564) q[1];
ry(1.2945827) q[2];
ry(0.22267312) q[3];
ryy_139854475745168(1.0205148) q[0],q[1];
ryy_139854475746752(-3.0007863) q[1],q[2];
ryy_139854475745264(0.28826877) q[2],q[3];
h q[0];
h q[1];
h q[2];
h q[3];
ry(-0.84834123) q[0];
ry(1.6489421) q[1];
ry(1.6005687) q[2];
ry(-0.078589246) q[3];
ryy_139854475222224(-1.3988656) q[0],q[1];
ryy_139854475222848(2.639245) q[1],q[2];
ryy_139854475219968(-0.12578748) q[2],q[3];
h q[0];
h q[1];
h q[2];
h q[3];
ry(0.43104586) q[0];
ry(0.36835304) q[1];
ry(0.7637977) q[2];
ry(1.1792222) q[3];
ryy_139854492587680(0.15877706) q[0],q[1];
ryy_139854492588304(0.28134722) q[1],q[2];
ryy_139854492588880(0.90068722) q[2],q[3];
rx(-1.3154298) q[0];
rx(0.010911906) q[1];
rx(0.41278124) q[2];
rx(0.57027513) q[3];
rz(0.93254125) q[0];
rz(1.4955928) q[1];
rz(2.0166564) q[2];
rz(-1.1767042) q[3];
rx(1.4340464) q[0];
rx(0.24152994) q[1];
rx(0.75475419) q[2];
rx(2.0993958) q[3];
rz(0.12709014) q[0];
rz(1.6027033e-07) q[1];
rz(-0.0059957178) q[2];
rz(4.738514e-10) q[3];
measure q[0] -> c[0];
measure q[1] -> c[1];
measure q[2] -> c[2];
measure q[3] -> c[3];
