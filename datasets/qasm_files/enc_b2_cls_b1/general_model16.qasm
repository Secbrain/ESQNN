OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
creg c[4];
ry(0.76641285) q[0];
ry(0.43247354) q[1];
ry(-1.0184042) q[2];
ry(0.012393372) q[3];
rz(0.66951525) q[0];
rz(1.2033629) q[1];
rz(0.82982934) q[2];
rz(-0.91916043) q[3];
rx(0.95823151) q[0];
rx(0.27241537) q[1];
rx(0.99157894) q[2];
rx(0.38257098) q[3];
ry(-0.50297844) q[0];
ry(1.5228251) q[1];
ry(-2.5021319) q[2];
ry(-0.62520558) q[3];
ry(-0.45196578) q[0];
ry(-0.57200813) q[1];
ry(1.7311065) q[2];
ry(0.40453166) q[3];
rz(0.59107059) q[0];
rz(0.5206778) q[1];
rz(0.26302585) q[2];
rz(1.4625489) q[3];
rx(0.80679321) q[0];
rx(2.106061) q[1];
rx(0.12502445) q[2];
rx(-1.0741942) q[3];
ry(-0.044601873) q[0];
ry(-0.77408612) q[1];
ry(1.4476148) q[2];
ry(-1.809449) q[3];
rx(-1.2092094) q[0];
rx(3.4802942) q[1];
rx(-1.6305592) q[2];
rx(-0.47582367) q[3];
rz(-0.0025946714) q[0];
rz(-0.14258796) q[1];
rz(-0.42338508) q[2];
rz(0.8877607) q[3];
crz(0.66178602) q[0],q[1];
crz(0.055840801) q[2],q[3];
crz(-0.0055234241) q[1],q[2];
measure q[0] -> c[0];
measure q[1] -> c[1];
measure q[2] -> c[2];
measure q[3] -> c[3];
