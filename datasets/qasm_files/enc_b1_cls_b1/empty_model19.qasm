OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
creg c[4];
rx(2.6646845) q[0];
rx(-2.792083) q[1];
rx(-2.3284972) q[2];
rx(-1.104159) q[3];
rz(0.21990129) q[0];
rz(-0.061742522) q[1];
rz(0.16701061) q[2];
rz(-2.6627817) q[3];
crx(1.4817467) q[0],q[3];
crx(-2.5448413) q[1],q[0];
crx(-2.8867974) q[2],q[1];
crx(-0.10476599) q[3],q[2];
measure q[0] -> c[0];
measure q[1] -> c[1];
measure q[2] -> c[2];
measure q[3] -> c[3];
