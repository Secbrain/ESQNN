OPENQASM 2.0;
include "qelib1.inc";
gate ryy(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(0.049137075) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
gate ryy_139907418907600(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(-0.073542528) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
gate ryy_139907418907120(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(0.64656734) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
gate ryy_139907418907216(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(-0.095311709) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
gate ryy_139907418909808(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(0.90738708) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
gate ryy_139907418909136(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(-0.92340708) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
gate ryy_139907418910240(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(0.11408052) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
gate ryy_139907418910096(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(-0.34072387) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
gate ryy_139907418909664(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(-2.4524713) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
gate ryy_139908090223968(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(0.011645038) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
gate ryy_139908090221856(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(0.13641059) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
gate ryy_139908090221328(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(0.55829668) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
qreg q[4];
creg c[4];
h q[0];
h q[1];
h q[2];
h q[3];
ry(-1.2786967) q[0];
ry(-0.038427468) q[1];
ry(1.913801) q[2];
ry(0.33784461) q[3];
ryy(0.049137075) q[0],q[1];
ryy_139907418907600(-0.073542528) q[1],q[2];
ryy_139907418907120(0.64656734) q[2],q[3];
h q[0];
h q[1];
h q[2];
h q[3];
ry(0.12505612) q[0];
ry(-0.76215148) q[1];
ry(-1.19056) q[2];
ry(0.77560735) q[3];
ryy_139907418907216(-0.095311709) q[0],q[1];
ryy_139907418909808(0.90738708) q[1],q[2];
ryy_139907418909136(-0.92340708) q[2],q[3];
h q[0];
h q[1];
h q[2];
h q[3];
ry(0.45571992) q[0];
ry(0.25033033) q[1];
ry(-1.3610971) q[2];
ry(1.8018341) q[3];
ryy_139907418910240(0.11408052) q[0],q[1];
ryy_139907418910096(-0.34072387) q[1],q[2];
ryy_139907418909664(-2.4524713) q[2],q[3];
h q[0];
h q[1];
h q[2];
h q[3];
ry(-0.074341945) q[0];
ry(-0.15664156) q[1];
ry(-0.8708455) q[2];
ry(-0.64109725) q[3];
ryy_139908090223968(0.011645038) q[0],q[1];
ryy_139908090221856(0.13641059) q[1],q[2];
ryy_139908090221328(0.55829668) q[2],q[3];
rx(0.33259735) q[0];
rx(0.077911101) q[1];
rx(3.170784) q[2];
rx(-3.8348358) q[3];
rz(0.5580616) q[0];
rz(0.62850559) q[1];
rz(-0.003112227) q[2];
rz(-0.25633994) q[3];
cx q[0],q[1];
cx q[1],q[2];
cx q[2],q[3];
measure q[0] -> c[0];
measure q[1] -> c[1];
measure q[2] -> c[2];
measure q[3] -> c[3];
