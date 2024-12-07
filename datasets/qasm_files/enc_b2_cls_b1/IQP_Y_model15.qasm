OPENQASM 2.0;
include "qelib1.inc";
gate ryy(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(-0.81859905) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
gate ryy_139858765311040(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(1.1764641) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
gate ryy_139858663742912(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(0.01003588) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
gate ryy_139858775974576(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(0.32540655) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
gate ryy_139858662983856(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(0.14194307) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
gate ryy_139858662981984(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(-0.13652836) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
gate ryy_139858765180832(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(-0.16256219) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
gate ryy_139858765178864(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(0.32747591) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
gate ryy_139858765176992(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(2.1587372) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
gate ryy_139858775946240(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(0.40895355) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
gate ryy_139858775949264(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(-0.35201019) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
gate ryy_139858775945808(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(-0.41088566) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
gate ryy_139858664031328(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(-0.84520173) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
gate ryy_139858664031376(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(0.53367859) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
gate ryy_139858664033392(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(0.67364782) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
gate ryy_139858775983440(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(0.27201954) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
gate ryy_139858775984160(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(-1.6335634) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
gate ryy_139858775985696(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(1.162483) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
gate ryy_139858663592672(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(0.12727796) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
gate ryy_139858765142000(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(0.429984) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
gate ryy_139858765143728(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(-0.27398485) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
gate ryy_139858765140512(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(0.67849571) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
gate ryy_139858765143440(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(1.6920573) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
gate ryy_139858765140320(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(-0.63436013) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
qreg q[4];
creg c[4];
h q[0];
h q[1];
h q[2];
h q[3];
ry(-0.79778224) q[0];
ry(1.0260934) q[1];
ry(1.1465468) q[2];
ry(0.0087531358) q[3];
ryy(-0.81859905) q[0],q[1];
ryy_139858765311040(1.1764641) q[1],q[2];
ryy_139858663742912(0.01003588) q[2],q[3];
h q[0];
h q[1];
h q[2];
h q[3];
ry(-0.30345869) q[0];
ry(-1.0723257) q[1];
ry(-0.13236935) q[2];
ry(1.0314196) q[3];
ryy_139858775974576(0.32540655) q[0],q[1];
ryy_139858662983856(0.14194307) q[1],q[2];
ryy_139858662981984(-0.13652836) q[2],q[3];
h q[0];
h q[1];
h q[2];
h q[3];
ry(-0.87215644) q[0];
ry(0.18639109) q[1];
ry(1.756929) q[2];
ry(1.2286991) q[3];
ryy_139858765180832(-0.16256219) q[0],q[1];
ryy_139858765178864(0.32747591) q[1],q[2];
ryy_139858765176992(2.1587372) q[2],q[3];
h q[0];
h q[1];
h q[2];
h q[3];
ry(0.59398216) q[0];
ry(0.68849468) q[1];
ry(-0.51127511) q[2];
ry(0.80364889) q[3];
ryy_139858775946240(0.40895355) q[0],q[1];
ryy_139858775949264(-0.35201019) q[1],q[2];
ryy_139858775945808(-0.41088566) q[2],q[3];
h q[0];
h q[1];
h q[2];
h q[3];
ry(1.5612283) q[0];
ry(-0.54136974) q[1];
ry(-0.98579317) q[2];
ry(-0.68335617) q[3];
ryy_139858664031328(-0.84520173) q[0],q[1];
ryy_139858664031376(0.53367859) q[1],q[2];
ryy_139858664033392(0.67364782) q[2],q[3];
h q[0];
h q[1];
h q[2];
h q[3];
ry(-0.32688567) q[0];
ry(-0.83215499) q[1];
ry(1.9630519) q[2];
ry(0.59218144) q[3];
ryy_139858775983440(0.27201954) q[0],q[1];
ryy_139858775984160(-1.6335634) q[1],q[2];
ryy_139858775985696(1.162483) q[2],q[3];
h q[0];
h q[1];
h q[2];
h q[3];
ry(-0.63652593) q[0];
ry(-0.19995722) q[1];
ry(-2.1503799) q[2];
ry(0.1274123) q[3];
ryy_139858663592672(0.12727796) q[0],q[1];
ryy_139858765142000(0.429984) q[1],q[2];
ryy_139858765143728(-0.27398485) q[2],q[3];
h q[0];
h q[1];
h q[2];
h q[3];
ry(0.37832859) q[0];
ry(1.7934031) q[1];
ry(0.94348961) q[2];
ry(-0.67235518) q[3];
ryy_139858765140512(0.67849571) q[0],q[1];
ryy_139858765143440(1.6920573) q[1],q[2];
ryy_139858765140320(-0.63436013) q[2],q[3];
ry(2.0354607) q[0];
ry(-0.42951801) q[1];
ry(-0.19233072) q[2];
ry(0.31648394) q[3];
cx q[0],q[3];
cx q[1],q[0];
cx q[2],q[1];
cx q[3],q[2];
ry(0.070207357) q[0];
ry(-0.65833282) q[1];
ry(-0.73005038) q[2];
ry(2.3802059) q[3];
cx q[0],q[1];
cx q[3],q[0];
cx q[2],q[3];
cx q[1],q[2];
measure q[0] -> c[0];
measure q[1] -> c[1];
measure q[2] -> c[2];
measure q[3] -> c[3];
