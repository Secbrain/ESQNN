OPENQASM 2.0;
include "qelib1.inc";
gate ryy(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(-0.15700151) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
gate ryy_139854475628448(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(0.24314609) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
gate ryy_139854475555936(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(0.54264176) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
gate ryy_139854492719616(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(0.15975812) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
gate ryy_139854475169744(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(0.56425166) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
gate ryy_139854475563648(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(0.73695487) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
gate ryy_139854520212688(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(-0.014871106) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
gate ryy_139854520212880(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(0.022446999) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
gate ryy_139854520213456(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(0.20508711) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
gate ryy_139854520209760(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(0.49215579) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
gate ryy_139854520211392(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(-0.59990662) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
gate ryy_139854520212832(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(-0.23361835) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
qreg q[4];
creg c[4];
h q[0];
h q[1];
h q[2];
h q[3];
ry(-0.22056575) q[0];
ry(0.71181273) q[1];
ry(0.34158716) q[2];
ry(1.5885895) q[3];
ryy(-0.15700151) q[0],q[1];
ryy_139854475628448(0.24314609) q[1],q[2];
ryy_139854475555936(0.54264176) q[2],q[3];
h q[0];
h q[1];
h q[2];
h q[3];
ry(-0.34887818) q[0];
ry(-0.45791951) q[1];
ry(-1.2322071) q[2];
ry(-0.59807712) q[3];
ryy_139854492719616(0.15975812) q[0],q[1];
ryy_139854475169744(0.56425166) q[1],q[2];
ryy_139854475563648(0.73695487) q[2],q[3];
h q[0];
h q[1];
h q[2];
h q[3];
ry(-0.28154701) q[0];
ry(0.052819263) q[1];
ry(0.42497751) q[2];
ry(0.48258343) q[3];
ryy_139854520212688(-0.014871106) q[0],q[1];
ryy_139854520212880(0.022446999) q[1],q[2];
ryy_139854520213456(0.20508711) q[2],q[3];
h q[0];
h q[1];
h q[2];
h q[3];
ry(0.48813388) q[0];
ry(1.0082394) q[1];
ry(-0.59500414) q[2];
ry(0.39263314) q[3];
ryy_139854520209760(0.49215579) q[0],q[1];
ryy_139854520211392(-0.59990662) q[1],q[2];
ryy_139854520212832(-0.23361835) q[2],q[3];
rx(-0.077625506) q[0];
rx(-0.15773045) q[1];
rx(-3.7177804) q[2];
rx(2.3816924) q[3];
rz(1.3131557) q[0];
rz(1.614441) q[1];
rz(-1.5993855) q[2];
rz(-0.11060566) q[3];
crx(3.0586119) q[0],q[1];
crx(1.5476652) q[2],q[3];
rx(-2.8477533) q[0];
rx(0.28494361) q[1];
rx(-3.209311) q[2];
rx(-2.3754127) q[3];
rz(-1.643407) q[0];
rz(1.4920843) q[1];
rz(2.3978894) q[2];
rz(-1.8637288) q[3];
crx(1.5103924) q[1],q[2];
rx(1.8839356) q[0];
rx(-2.0950108) q[1];
rx(0.40572515) q[2];
rx(2.2213643) q[3];
rz(2.947) q[0];
rz(2.6973715) q[1];
rz(-1.9528952) q[2];
rz(2.3241549) q[3];
crx(-0.96307349) q[0],q[1];
crx(1.6177337) q[2],q[3];
rx(-0.20550804) q[0];
rx(-1.5195923) q[1];
rx(1.452499) q[2];
rx(0.73448557) q[3];
rz(-0.47305688) q[0];
rz(-3.1527063e-07) q[1];
rz(1.7580252) q[2];
rz(-1.3445501e-09) q[3];
crx(-2.7027135) q[1],q[2];
measure q[0] -> c[0];
measure q[1] -> c[1];
measure q[2] -> c[2];
measure q[3] -> c[3];
