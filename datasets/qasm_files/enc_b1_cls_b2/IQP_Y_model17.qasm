OPENQASM 2.0;
include "qelib1.inc";
gate ryy(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(-0.00040878361) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
gate ryy_139854474884288(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(-0.25196332) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
gate ryy_139854474884720(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(0.8114062) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
gate ryy_139854474883952(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(-0.36726758) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
gate ryy_139854474886064(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(0.31130195) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
gate ryy_139854474886352(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(-1.6615695) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
gate ryy_139854474886736(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(0.28779781) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
gate ryy_139854474885008(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(-0.055162787) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
gate ryy_139854474883664(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(-0.011662201) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
gate ryy_139854492669552(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(0.045743082) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
gate ryy_139854492669888(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(-0.0026256645) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
gate ryy_139854492668112(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(-0.22273271) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
qreg q[4];
creg c[4];
h q[0];
h q[1];
h q[2];
h q[3];
ry(0.0019485307) q[0];
ry(-0.20979071) q[1];
ry(1.2010223) q[2];
ry(0.6755963) q[3];
ryy(-0.00040878361) q[0],q[1];
ryy_139854474884288(-0.25196332) q[1],q[2];
ryy_139854474884720(0.8114062) q[2],q[3];
h q[0];
h q[1];
h q[2];
h q[3];
ry(-1.8900177) q[0];
ry(0.19431965) q[1];
ry(1.6020095) q[2];
ry(-1.0371783) q[3];
ryy_139854474883952(-0.36726758) q[0],q[1];
ryy_139854474886064(0.31130195) q[1],q[2];
ryy_139854474886352(-1.6615695) q[2],q[3];
h q[0];
h q[1];
h q[2];
h q[3];
ry(-0.74868762) q[0];
ry(-0.38440305) q[1];
ry(0.14350247) q[2];
ry(-0.081268296) q[3];
ryy_139854474886736(0.28779781) q[0],q[1];
ryy_139854474885008(-0.055162787) q[1],q[2];
ryy_139854474883664(-0.011662201) q[2],q[3];
h q[0];
h q[1];
h q[2];
h q[3];
ry(1.1261654) q[0];
ry(0.040618442) q[1];
ry(-0.064642176) q[2];
ry(3.4456251) q[3];
ryy_139854492669552(0.045743082) q[0],q[1];
ryy_139854492669888(-0.0026256645) q[1],q[2];
ryy_139854492668112(-0.22273271) q[2],q[3];
rx(1.0882084) q[0];
rx(-1.5828558) q[1];
rx(2.2191613) q[2];
rx(2.7519643) q[3];
rz(2.2553875) q[0];
rz(-2.3851962) q[1];
rz(3.9513519) q[2];
rz(-2.3445537) q[3];
crz(-1.391209) q[0],q[1];
crz(-3.2042117) q[2],q[3];
rx(2.4830782) q[0];
rx(-2.3661506) q[1];
rx(0.60889053) q[2];
rx(0.71713412) q[3];
rz(-0.17950003) q[0];
rz(-0.38971534) q[1];
rz(-0.25889224) q[2];
rz(0.71466488) q[3];
crz(1.1114719) q[1],q[2];
rx(2.0342462) q[0];
rx(1.2404382) q[1];
rx(-1.745574) q[2];
rx(1.9355228) q[3];
rz(1.9108125) q[0];
rz(0.86071509) q[1];
rz(-1.6380943) q[2];
rz(-2.5620227) q[3];
crz(1.97921) q[0],q[1];
crz(2.8718746) q[2],q[3];
rx(2.8923752) q[0];
rx(1.3315192) q[1];
rx(-2.0329661) q[2];
rx(2.2025852) q[3];
rz(0.92023605) q[0];
rz(-2.5451776e-09) q[1];
rz(-0.53516185) q[2];
rz(0.16923253) q[3];
crz(0.019033289) q[1],q[2];
measure q[0] -> c[0];
measure q[1] -> c[1];
measure q[2] -> c[2];
measure q[3] -> c[3];
