OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
creg c[4];
ry(3.5289621) q[0];
ry(-0.50436157) q[1];
ry(2.1838768) q[2];
ry(0.10185775) q[3];
crz(-0.41461217) q[0],q[3];
crz(-1.0207587) q[1],q[0];
crz(3.0014215) q[2],q[1];
crz(-0.58403808) q[3],q[2];
ry(1.8195595) q[0];
ry(-0.21398988) q[1];
ry(0.049063873) q[2];
ry(2.8678515) q[3];
crz(0.0013047332) q[0],q[1];
crz(-0.9379245) q[3],q[0];
crz(2.0529435e-06) q[2],q[3];
crz(-4.739042e-09) q[1],q[2];
measure q[0] -> c[0];
measure q[1] -> c[1];
measure q[2] -> c[2];
measure q[3] -> c[3];
