OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
creg c[4];
h q[0];
h q[1];
h q[2];
h q[3];
rz(-1.0562596) q[0];
rz(0.24130821) q[1];
rz(0.1827537) q[2];
rz(0.62465245) q[3];
rzz(-0.25488412) q[0],q[1];
rzz(0.044099968) q[1],q[2];
rzz(0.11415754) q[2],q[3];
h q[0];
h q[1];
h q[2];
h q[3];
rz(-0.79397756) q[0];
rz(-0.67483521) q[1];
rz(-0.38768774) q[2];
rz(0.44965044) q[3];
rzz(0.53580403) q[0],q[1];
rzz(0.26162535) q[1],q[2];
rzz(-0.17432396) q[2],q[3];
h q[0];
h q[1];
h q[2];
h q[3];
rz(0.37261015) q[0];
rz(-1.9104947) q[1];
rz(0.26085028) q[2];
rz(1.4177611) q[3];
rzz(-0.71186972) q[0],q[1];
rzz(-0.49835306) q[1],q[2];
rzz(0.36982337) q[2],q[3];
h q[0];
h q[1];
h q[2];
h q[3];
rz(0.67380321) q[0];
rz(1.4665507) q[1];
rz(-1.1077474) q[2];
rz(-0.74437821) q[3];
rzz(0.98816657) q[0],q[1];
rzz(-1.6245677) q[1],q[2];
rzz(0.82458305) q[2],q[3];
h q[0];
h q[1];
h q[2];
h q[3];
cz q[0],q[1];
cz q[1],q[2];
cz q[2],q[3];
rx(2.9422672) q[0];
rx(-3.5090039) q[1];
rx(-3.2150812) q[2];
rx(-2.4479613) q[3];
measure q[0] -> c[0];
measure q[1] -> c[1];
measure q[2] -> c[2];
measure q[3] -> c[3];
