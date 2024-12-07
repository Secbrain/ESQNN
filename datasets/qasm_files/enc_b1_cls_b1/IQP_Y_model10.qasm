OPENQASM 2.0;
include "qelib1.inc";
gate ryy(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(0.27496207) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
gate ryy_139907351575136(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(-0.37036154) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
gate ryy_139907351576480(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(-0.78784591) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
gate ryy_139908160043712(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(-0.77043146) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
gate ryy_139908160043184(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(0.60433972) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
gate ryy_139908160043232(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(2.0709898) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
gate ryy_139908160041552(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(0.38665238) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
gate ryy_139908160041888(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(-0.12017315) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
gate ryy_139908160042368(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(-0.038478117) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
gate ryy_139908160042704(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(0.85190964) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
gate ryy_139908160041216(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(2.6598103) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
gate ryy_139908160041792(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(-0.66262156) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
qreg q[4];
creg c[4];
h q[0];
h q[1];
h q[2];
h q[3];
ry(-0.71110868) q[0];
ry(-0.38666672) q[1];
ry(0.95783144) q[2];
ry(-0.82253087) q[3];
ryy(0.27496207) q[0],q[1];
ryy_139907351575136(-0.37036154) q[1],q[2];
ryy_139907351576480(-0.78784591) q[2],q[3];
h q[0];
h q[1];
h q[2];
h q[3];
ry(-2.390805) q[0];
ry(0.32224771) q[1];
ry(1.8753887) q[2];
ry(1.1042989) q[3];
ryy_139908160043712(-0.77043146) q[0],q[1];
ryy_139908160043184(0.60433972) q[1],q[2];
ryy_139908160043232(2.0709898) q[2],q[3];
h q[0];
h q[1];
h q[2];
h q[3];
ry(-0.52237588) q[0];
ry(-0.74018037) q[1];
ry(0.16235657) q[2];
ry(-0.2369976) q[3];
ryy_139908160041552(0.38665238) q[0],q[1];
ryy_139908160041888(-0.12017315) q[1],q[2];
ryy_139908160042368(-0.038478117) q[2],q[3];
h q[0];
h q[1];
h q[2];
h q[3];
ry(0.50993472) q[0];
ry(1.6706249) q[1];
ry(1.592105) q[2];
ry(-0.41619211) q[3];
ryy_139908160042704(0.85190964) q[0],q[1];
ryy_139908160041216(2.6598103) q[1],q[2];
ryy_139908160041792(-0.66262156) q[2],q[3];
ry(-1.1159531) q[0];
ry(-3.3346825) q[1];
ry(1.3278183) q[2];
ry(-1.6478095) q[3];
cz q[0],q[1];
cz q[1],q[2];
cz q[2],q[3];
cz q[3],q[0];
ry(-0.53522933) q[0];
ry(3.198884) q[1];
ry(4.4147143) q[2];
ry(-3.0191317) q[3];
measure q[0] -> c[0];
measure q[1] -> c[1];
measure q[2] -> c[2];
measure q[3] -> c[3];
