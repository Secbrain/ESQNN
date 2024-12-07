OPENQASM 2.0;
include "qelib1.inc";
gate ryy(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(1.176784) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
gate ryy_139854475046672(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(3.5842724) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
gate ryy_139854475046096(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(1.1023446) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
gate ryy_139854474962304(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(-0.14602424) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
gate ryy_139854474963984(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(-0.19829972) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
gate ryy_139854474962448(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(-1.1768631) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
gate ryy_139854474932720(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(0.33402631) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
gate ryy_139854474932768(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(-0.7547828) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
gate ryy_139854474933104(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(1.6372216) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
gate ryy_139854520178576(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(0.092073612) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
gate ryy_139854474933488(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(0.076018088) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
gate ryy_139854474934256(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(-1.4290465) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
qreg q[4];
creg c[4];
h q[0];
h q[1];
h q[2];
h q[3];
ry(-0.5452711) q[0];
ry(-2.1581633) q[1];
ry(-1.6607976) q[2];
ry(-0.66374415) q[3];
ryy(1.176784) q[0],q[1];
ryy_139854475046672(3.5842724) q[1],q[2];
ryy_139854475046096(1.1023446) q[2],q[3];
h q[0];
h q[1];
h q[2];
h q[3];
ry(0.36579075) q[0];
ry(-0.39920157) q[1];
ry(0.49674082) q[2];
ry(-2.3691692) q[3];
ryy_139854474962304(-0.14602424) q[0],q[1];
ryy_139854474963984(-0.19829972) q[1],q[2];
ryy_139854474962448(-1.1768631) q[2],q[3];
h q[0];
h q[1];
h q[2];
h q[3];
ry(-0.56147081) q[0];
ry(-0.59491307) q[1];
ry(1.2687279) q[2];
ry(1.2904434) q[3];
ryy_139854474932720(0.33402631) q[0],q[1];
ryy_139854474932768(-0.7547828) q[1],q[2];
ryy_139854474933104(1.6372216) q[2],q[3];
h q[0];
h q[1];
h q[2];
h q[3];
ry(-1.1755682) q[0];
ry(-0.078322642) q[1];
ry(-0.97057611) q[2];
ry(1.4723693) q[3];
ryy_139854520178576(0.092073612) q[0],q[1];
ryy_139854474933488(0.076018088) q[1],q[2];
ryy_139854474934256(-1.4290465) q[2],q[3];
h q[0];
h q[1];
h q[2];
h q[3];
cz q[0],q[1];
cz q[1],q[2];
cz q[2],q[3];
rx(-1.441366) q[0];
rx(-1.5748801) q[1];
rx(1.1854918) q[2];
rx(1.7294676) q[3];
h q[0];
h q[1];
h q[2];
h q[3];
cz q[0],q[1];
cz q[1],q[2];
cz q[2],q[3];
rx(1.07974) q[0];
rx(4.4267321) q[1];
rx(2.6858153) q[2];
rx(0.60093248) q[3];
measure q[0] -> c[0];
measure q[1] -> c[1];
measure q[2] -> c[2];
measure q[3] -> c[3];
