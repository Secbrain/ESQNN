OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
creg c[4];
rx(-1.4078194) q[0];
rx(-0.080110811) q[1];
rx(0.51941246) q[2];
rx(1.1708889) q[3];
ry(2.1779797) q[0];
ry(1.7791979) q[1];
ry(0.25832492) q[2];
ry(-2.4340737) q[3];
rz(-0.34975004) q[0];
rz(-1.338056) q[1];
rz(-0.43891034) q[2];
rz(-0.58501744) q[3];
rx(1.8071492) q[0];
rx(-0.73262411) q[1];
rx(0.40939674) q[2];
rx(-0.58409548) q[3];
rx(4.2621083) q[0];
rx(-5.0847483) q[1];
rx(-1.9226303) q[2];
rx(-1.9590893) q[3];
rz(0.55733818) q[0];
rz(0.16685125) q[1];
rz(0.072726183) q[2];
rz(-0.37133586) q[3];
crz(-1.5721517e-10) q[0],q[3];
crz(0.011661674) q[1],q[0];
crz(-1.0090107e-10) q[2],q[1];
crz(0.43422297) q[3],q[2];
measure q[0] -> c[0];
measure q[1] -> c[1];
measure q[2] -> c[2];
measure q[3] -> c[3];