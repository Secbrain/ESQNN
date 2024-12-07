OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
creg c[4];
rx(1.6426471) q[0];
rx(0.61727381) q[1];
rx(1.7688652) q[2];
rx(2.9097929) q[3];
rz(2.1428111) q[0];
rz(0.57796514) q[1];
rz(-1.6990906) q[2];
rz(0.049253047) q[3];
crz(2.5814297) q[0],q[1];
crz(-0.47805834) q[0],q[2];
crz(0.42260897) q[0],q[3];
crz(2.5154259) q[1],q[0];
crz(1.186313) q[1],q[2];
crz(0.48751086) q[1],q[3];
crz(0.20387258) q[2],q[0];
crz(-1.0860082) q[2],q[1];
crz(1.8861651) q[2],q[3];
crz(1.7824517) q[3],q[0];
crz(-1.3959469) q[3],q[1];
crz(0.4018971) q[3],q[2];
rx(0.96210641) q[0];
rx(0.80583525) q[1];
rx(1.0101322) q[2];
rx(-0.65079218) q[3];
rz(-0.45433834) q[0];
rz(0.28768381) q[1];
rz(0.1576701) q[2];
rz(-0.0023989542) q[3];
measure q[0] -> c[0];
measure q[1] -> c[1];
measure q[2] -> c[2];
measure q[3] -> c[3];
