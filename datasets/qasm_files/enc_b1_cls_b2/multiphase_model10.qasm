OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
creg c[4];
rx(-0.71110868) q[0];
rx(-0.38666672) q[1];
rx(0.95783144) q[2];
rx(-0.82253087) q[3];
ry(-2.390805) q[0];
ry(0.32224771) q[1];
ry(1.8753887) q[2];
ry(1.1042989) q[3];
rz(-0.52237588) q[0];
rz(-0.74018037) q[1];
rz(0.16235657) q[2];
rz(-0.2369976) q[3];
rx(0.50993472) q[0];
rx(1.6706249) q[1];
rx(1.592105) q[2];
rx(-0.41619211) q[3];
ry(-2.0912983) q[0];
ry(-2.267633) q[1];
ry(2.6046908) q[2];
ry(1.9266528) q[3];
cz q[0],q[1];
cz q[1],q[2];
cz q[2],q[3];
cz q[3],q[0];
ry(-0.97811288) q[0];
ry(2.0730755) q[1];
ry(2.0207169) q[2];
ry(-1.3863395) q[3];
ry(2.4886956) q[0];
ry(-1.504436) q[1];
ry(0.66172904) q[2];
ry(1.4116864) q[3];
cz q[0],q[1];
cz q[1],q[2];
cz q[2],q[3];
cz q[3],q[0];
ry(0.015012771) q[0];
ry(0.16293354) q[1];
ry(-3.0005074) q[2];
ry(-0.69852084) q[3];
measure q[0] -> c[0];
measure q[1] -> c[1];
measure q[2] -> c[2];
measure q[3] -> c[3];
