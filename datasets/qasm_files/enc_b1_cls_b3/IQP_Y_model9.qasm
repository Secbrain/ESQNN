OPENQASM 2.0;
include "qelib1.inc";
gate ryy(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(0.27095616) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
gate ryy_140342689871856(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(-0.230354) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
gate ryy_140342689872960(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(0.081311814) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
gate ryy_140342689870608(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(1.5190563) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
gate ryy_140342689870272(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(2.5725815) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
gate ryy_140342689871328(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(-1.4086758) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
gate ryy_140342690007552(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(0.0035306734) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
gate ryy_140342690007696(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(-0.22907366) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
gate ryy_140342690007648(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(-0.55479783) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
gate ryy_140342689749984(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(-0.21167442) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
gate ryy_140342689748064(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(0.30889323) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
gate ryy_140342689750080(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(-1.5864727) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
qreg q[4];
creg c[4];
h q[0];
h q[1];
h q[2];
h q[3];
ry(-0.56102747) q[0];
ry(-0.48296416) q[1];
ry(0.47695878) q[2];
ry(0.17047974) q[3];
ryy(0.27095616) q[0],q[1];
ryy_140342689871856(-0.230354) q[1],q[2];
ryy_140342689872960(0.081311814) q[2],q[3];
h q[0];
h q[1];
h q[2];
h q[3];
ry(0.8450036) q[0];
ry(1.7976921) q[1];
ry(1.4310468) q[2];
ry(-0.98436731) q[3];
ryy_140342689870608(1.5190563) q[0],q[1];
ryy_140342689870272(2.5725815) q[1],q[2];
ryy_140342689871328(-1.4086758) q[2],q[3];
h q[0];
h q[1];
h q[2];
h q[3];
ry(0.011664378) q[0];
ry(0.30268854) q[1];
ry(-0.7567966) q[2];
ry(0.73308712) q[3];
ryy_140342690007552(0.0035306734) q[0],q[1];
ryy_140342690007696(-0.22907366) q[1],q[2];
ryy_140342690007648(-0.55479783) q[2],q[3];
h q[0];
h q[1];
h q[2];
h q[3];
ry(-1.0032725) q[0];
ry(0.21098398) q[1];
ry(1.4640602) q[2];
ry(-1.0836117) q[3];
ryy_140342689749984(-0.21167442) q[0],q[1];
ryy_140342689748064(0.30889323) q[1],q[2];
ryy_140342689750080(-1.5864727) q[2],q[3];
h q[0];
h q[1];
h q[2];
h q[3];
cz q[0],q[1];
cz q[1],q[2];
cz q[2],q[3];
rx(0.22386666) q[0];
rx(-4.1045771) q[1];
rx(-2.6124315) q[2];
rx(0.7514832) q[3];
h q[0];
h q[1];
h q[2];
h q[3];
cz q[0],q[1];
cz q[1],q[2];
cz q[2],q[3];
rx(-2.0017948) q[0];
rx(2.257683) q[1];
rx(-3.1296539) q[2];
rx(2.3395898) q[3];
h q[0];
h q[1];
h q[2];
h q[3];
cz q[0],q[1];
cz q[1],q[2];
cz q[2],q[3];
rx(2.3709702) q[0];
rx(1.8549963) q[1];
rx(1.6403663) q[2];
rx(1.2413065) q[3];
measure q[0] -> c[0];
measure q[1] -> c[1];
measure q[2] -> c[2];
measure q[3] -> c[3];
