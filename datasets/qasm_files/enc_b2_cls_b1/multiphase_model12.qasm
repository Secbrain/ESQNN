OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
creg c[4];
rx(-1.0025995) q[0];
rx(-0.86913598) q[1];
rx(1.0349028) q[2];
rx(1.1413633) q[3];
ry(-0.61135471) q[0];
ry(0.56699115) q[1];
ry(-0.15298273) q[2];
ry(-0.29165849) q[3];
rz(-1.2069418) q[0];
rz(-0.16841801) q[1];
rz(-1.0213231) q[2];
rz(0.45474377) q[3];
rx(0.056282237) q[0];
rx(0.19084698) q[1];
rx(-0.0028136375) q[2];
rx(-0.64237928) q[3];
rx(-0.23483104) q[0];
rx(0.18347847) q[1];
rx(0.82709885) q[2];
rx(0.68174583) q[3];
ry(0.40630636) q[0];
ry(1.706159) q[1];
ry(1.1659429) q[2];
ry(-0.24008061) q[3];
rz(0.22484587) q[0];
rz(-2.3762155) q[1];
rz(0.4015539) q[2];
rz(-2.2946165) q[3];
rx(0.9543997) q[0];
rx(-0.3883369) q[1];
rx(2.1959841) q[2];
rx(0.84125185) q[3];
ry(0.56040823) q[0];
ry(1.068222) q[1];
ry(1.1268125) q[2];
ry(1.0537009) q[3];
rz(0.1273317) q[0];
rz(-3.5349824) q[1];
rz(1.0502591) q[2];
rz(-0.0013386151) q[3];
cz q[0],q[1];
cz q[2],q[3];
ry(0.71365601) q[1];
ry(-2.0999413) q[2];
rz(-0.71707737) q[1];
rz(0.028854331) q[2];
cz q[1],q[2];
measure q[0] -> c[0];
measure q[1] -> c[1];
measure q[2] -> c[2];
measure q[3] -> c[3];
