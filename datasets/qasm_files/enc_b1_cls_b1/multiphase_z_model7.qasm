OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
creg c[4];
rz(-0.95563322) q[0];
rz(-0.77797991) q[1];
rz(0.6935069) q[2];
rz(-0.43586132) q[3];
rx(-0.9824549) q[0];
rx(-0.90628791) q[1];
rx(1.2595613) q[2];
rx(0.34626761) q[3];
ry(-2.0042973) q[0];
ry(0.0055122636) q[1];
ry(0.46033239) q[2];
ry(1.1791135) q[3];
rz(-1.010552) q[0];
rz(-0.62025863) q[1];
rz(-1.4548781) q[2];
rz(-0.58975685) q[3];
rx(-1.040157) q[0];
rx(-0.61811823) q[1];
rx(-1.7969543) q[2];
rx(1.9575455) q[3];
rz(1.048831) q[0];
rz(1.0555832) q[1];
rz(-0.49121216) q[2];
rz(2.1832864) q[3];
crz(-1.1384739) q[0],q[1];
crz(3.9920459) q[2],q[3];
rx(-1.2356814) q[0];
rx(-0.4264906) q[1];
rx(-2.7435069) q[2];
rx(-0.82951391) q[3];
rz(0.043820929) q[0];
rz(0.52246165) q[1];
rz(-0.00063713721) q[2];
rz(0.76164627) q[3];
crz(-0.013616901) q[1],q[2];
measure q[0] -> c[0];
measure q[1] -> c[1];
measure q[2] -> c[2];
measure q[3] -> c[3];
