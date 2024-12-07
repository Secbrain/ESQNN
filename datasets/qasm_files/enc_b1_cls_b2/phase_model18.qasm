OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
creg c[4];
rx(-1.7376028) q[0];
rx(-0.12535162) q[1];
rx(-1.3658148) q[2];
rx(1.1117461) q[3];
rx(-0.62279665) q[0];
rx(-0.78918087) q[1];
rx(-0.16782393) q[2];
rx(1.6433146) q[3];
rx(2.0070879) q[0];
rx(-1.2531019) q[1];
rx(1.1188694) q[2];
rx(1.7732776) q[3];
rx(-2.0716603) q[0];
rx(-0.41252553) q[1];
rx(-0.97695559) q[2];
rx(-0.033633888) q[3];
rx(-0.065097511) q[0];
rx(-0.15824759) q[1];
rx(-1.8618461) q[2];
rx(1.5522333) q[3];
rz(-3.1337392) q[0];
rz(0.043838851) q[1];
rz(-1.19168) q[2];
rz(-0.16057685) q[3];
crz(1.8971702) q[0],q[3];
crz(1.9511242) q[1],q[0];
crz(3.6219583) q[2],q[1];
crz(-0.17618614) q[3],q[2];
rx(-0.77672595) q[0];
rx(0.18800378) q[1];
rx(-2.0928054) q[2];
rx(-2.7787116) q[3];
rz(-6.354972e-07) q[0];
rz(-6.7848686e-07) q[1];
rz(0.18605518) q[2];
rz(-2.1307908e-07) q[3];
crz(2.2483286e-09) q[0],q[3];
crz(-0.0037630503) q[1],q[0];
crz(0.57359493) q[2],q[1];
crz(0.67563564) q[3],q[2];
measure q[0] -> c[0];
measure q[1] -> c[1];
measure q[2] -> c[2];
measure q[3] -> c[3];
