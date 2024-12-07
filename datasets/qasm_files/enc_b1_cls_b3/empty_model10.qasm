OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
creg c[4];
ry(0.79693818) q[0];
ry(1.9671099) q[1];
ry(-1.5692759) q[2];
ry(0.74290991) q[3];
cz q[0],q[1];
cz q[1],q[2];
cz q[2],q[3];
cz q[3],q[0];
ry(-0.78159183) q[0];
ry(0.29635343) q[1];
ry(-1.1818414) q[2];
ry(-2.5461977) q[3];
ry(0.77059418) q[0];
ry(-1.4183903) q[1];
ry(-1.8171153) q[2];
ry(-0.82435328) q[3];
cz q[0],q[1];
cz q[1],q[2];
cz q[2],q[3];
cz q[3],q[0];
ry(-0.57781172) q[0];
ry(-0.12690844) q[1];
ry(-0.29404747) q[2];
ry(-0.44276041) q[3];
ry(-2.3709204) q[0];
ry(2.0673945) q[1];
ry(0.31629127) q[2];
ry(-1.8440518) q[3];
cz q[0],q[1];
cz q[1],q[2];
cz q[2],q[3];
cz q[3],q[0];
ry(-2.1079452) q[0];
ry(-0.75748497) q[1];
ry(-1.8374312) q[2];
ry(2.5113263) q[3];
measure q[0] -> c[0];
measure q[1] -> c[1];
measure q[2] -> c[2];
measure q[3] -> c[3];
