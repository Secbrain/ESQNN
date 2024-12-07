OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
creg c[4];
h q[0];
h q[1];
h q[2];
h q[3];
rx(-1.0025995) q[0];
rx(-0.86913598) q[1];
rx(1.0349028) q[2];
rx(1.1413633) q[3];
rxx(0.87139529) q[0],q[1];
rxx(-0.89947128) q[1],q[2];
rxx(1.1812) q[2],q[3];
h q[0];
h q[1];
h q[2];
h q[3];
rx(-0.61135471) q[0];
rx(0.56699115) q[1];
rx(-0.15298273) q[2];
rx(-0.29165849) q[3];
rxx(-0.34663272) q[0],q[1];
rxx(-0.086739853) q[1],q[2];
rxx(0.044618711) q[2],q[3];
h q[0];
h q[1];
h q[2];
h q[3];
rx(-1.2069418) q[0];
rx(-0.16841801) q[1];
rx(-1.0213231) q[2];
rx(0.45474377) q[3];
rxx(0.20327073) q[0],q[1];
rxx(0.1720092) q[1],q[2];
rxx(-0.46444032) q[2],q[3];
h q[0];
h q[1];
h q[2];
h q[3];
rx(0.056282237) q[0];
rx(0.19084698) q[1];
rx(-0.0028136375) q[2];
rx(-0.64237928) q[3];
rxx(0.010741295) q[0],q[1];
rxx(-0.00053697423) q[1],q[2];
rxx(0.0018074225) q[2],q[3];
rx(-2.7687843) q[0];
rx(-2.0950477) q[1];
rx(-0.19796282) q[2];
rx(-3.1242688) q[3];
rz(1.3503851) q[0];
rz(2.8981609) q[1];
rz(-0.8159951) q[2];
rz(2.2164555) q[3];
crx(1.6770118) q[0],q[1];
crx(1.6002661) q[1],q[2];
crx(0.93267876) q[2],q[3];
rx(0.7330035) q[0];
rx(0.10841957) q[1];
rx(-0.011048467) q[2];
rx(1.6557707) q[3];
rz(-2.3665776) q[0];
rz(-2.5369909) q[1];
rz(0.55345583) q[2];
rz(-2.8413141) q[3];
crx(-0.50386101) q[0],q[1];
crx(0.96188504) q[1],q[2];
crx(-2.6290255) q[2],q[3];
rx(2.2185538) q[0];
rx(-1.7985314) q[1];
rx(-1.6978002) q[2];
rx(1.0314734) q[3];
rz(0.34149179) q[0];
rz(0.45111838) q[1];
rz(0.71107155) q[2];
rz(1.2492325) q[3];
crx(-0.38288057) q[0],q[1];
crx(-2.1310341) q[1],q[2];
crx(-1.8644991) q[2],q[3];
measure q[0] -> c[0];
measure q[1] -> c[1];
measure q[2] -> c[2];
measure q[3] -> c[3];
