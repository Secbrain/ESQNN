OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
creg c[4];
rz(0.28125459) q[0];
rz(0.087001391) q[1];
rz(-0.2570661) q[2];
rz(2.2180262) q[3];
rx(1.2401545) q[0];
rx(-0.65734249) q[1];
rx(1.8484452) q[2];
rx(-1.1966158) q[3];
ry(-0.45390239) q[0];
ry(1.4244479) q[1];
ry(2.2691953) q[2];
ry(1.3104836) q[3];
rz(-0.31789434) q[0];
rz(-0.37736565) q[1];
rz(2.2604442) q[2];
rz(-0.33095151) q[3];
h q[0];
h q[1];
h q[2];
h q[3];
cz q[0],q[1];
cz q[1],q[2];
cz q[2],q[3];
rx(-1.0361658) q[0];
rx(-1.3301682) q[1];
rx(1.8155977) q[2];
rx(-0.20478174) q[3];
h q[0];
h q[1];
h q[2];
h q[3];
cz q[0],q[1];
cz q[1],q[2];
cz q[2],q[3];
rx(2.1316853) q[0];
rx(4.7762666) q[1];
rx(3.0587056) q[2];
rx(-0.90402663) q[3];
h q[0];
h q[1];
h q[2];
h q[3];
cz q[0],q[1];
cz q[1],q[2];
cz q[2],q[3];
rx(-0.15688957) q[0];
rx(1.5960959) q[1];
rx(1.5714182) q[2];
rx(0.4189398) q[3];
measure q[0] -> c[0];
measure q[1] -> c[1];
measure q[2] -> c[2];
measure q[3] -> c[3];
