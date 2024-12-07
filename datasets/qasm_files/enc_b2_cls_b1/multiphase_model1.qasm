OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
creg c[4];
rx(-0.38097349) q[0];
rx(-1.6328287) q[1];
rx(-0.32575366) q[2];
rx(2.110002) q[3];
ry(-0.55823845) q[0];
ry(0.38789943) q[1];
ry(-2.2768142) q[2];
ry(0.37306783) q[3];
rz(0.20611548) q[0];
rz(-1.0159707) q[1];
rz(-0.59985167) q[2];
rz(0.17797667) q[3];
rx(-1.1433092) q[0];
rx(1.339462) q[1];
rx(-0.99675214) q[2];
rx(-0.60219049) q[3];
rx(0.87788355) q[0];
rx(0.7318399) q[1];
rx(0.27381453) q[2];
rx(1.7188367) q[3];
ry(1.3450311) q[0];
ry(-1.2653104) q[1];
ry(0.45117414) q[2];
ry(0.65972114) q[3];
rz(-0.65245253) q[0];
rz(-0.88792747) q[1];
rz(-1.0803741) q[2];
rz(1.4245183) q[3];
rx(-1.1213254) q[0];
rx(0.39183724) q[1];
rx(-0.49036843) q[2];
rx(-0.28906295) q[3];
rx(0.13481945) q[0];
rx(-0.43482682) q[1];
rx(1.8141373) q[2];
rx(-1.0760875) q[3];
rz(-0.0054390123) q[0];
rz(-0.056760889) q[1];
rz(1.0785027e-09) q[2];
rz(-0.89823931) q[3];
measure q[0] -> c[0];
measure q[1] -> c[1];
measure q[2] -> c[2];
measure q[3] -> c[3];