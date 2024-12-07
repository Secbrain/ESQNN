OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
creg c[4];
rx(-1.3846737) q[0];
rx(-0.87123615) q[1];
rx(-0.22336592) q[2];
rx(1.7173615) q[3];
rx(0.31888032) q[0];
rx(-0.42451897) q[1];
rx(0.30572093) q[2];
rx(-0.77459252) q[3];
rx(-1.5575725) q[0];
rx(0.99563611) q[1];
rx(-0.87978584) q[2];
rx(-0.60114205) q[3];
rx(-1.2741512) q[0];
rx(2.1227851) q[1];
rx(-1.2346531) q[2];
rx(-0.48791388) q[3];
rx(4.9178076) q[0];
rx(-3.1169157) q[1];
rx(0.37984571) q[2];
rx(-0.89311945) q[3];
rz(-0.13258992) q[0];
rz(-0.81850106) q[1];
rz(0.009819054) q[2];
rz(-3.5975555e-07) q[3];
cx q[0],q[1];
cx q[1],q[2];
cx q[2],q[3];
measure q[0] -> c[0];
measure q[1] -> c[1];
measure q[2] -> c[2];
measure q[3] -> c[3];
