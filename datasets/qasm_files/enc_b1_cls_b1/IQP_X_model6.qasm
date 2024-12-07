OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
creg c[4];
h q[0];
h q[1];
h q[2];
h q[3];
rx(-1.9005532) q[0];
rx(0.22857653) q[1];
rx(0.024859404) q[2];
rx(-0.34595025) q[3];
rxx(-0.43442187) q[0],q[1];
rxx(0.0056822761) q[1],q[2];
rxx(-0.0086001167) q[2],q[3];
h q[0];
h q[1];
h q[2];
h q[3];
rx(0.28683281) q[0];
rx(-0.73084241) q[1];
rx(0.17482026) q[2];
rx(-1.0939293) q[3];
rxx(-0.20962958) q[0],q[1];
rxx(-0.12776606) q[1],q[2];
rxx(-0.191241) q[2],q[3];
h q[0];
h q[1];
h q[2];
h q[3];
rx(-1.6021603) q[0];
rx(1.3528969) q[1];
rx(1.2888277) q[2];
rx(0.052295472) q[3];
rxx(-2.1675577) q[0],q[1];
rxx(1.743651) q[1],q[2];
rxx(0.067399852) q[2],q[3];
h q[0];
h q[1];
h q[2];
h q[3];
rx(-1.5468504) q[0];
rx(0.75670606) q[1];
rx(0.77551949) q[2];
rx(2.0265355) q[3];
rxx(-1.1705111) q[0],q[1];
rxx(0.58684027) q[1],q[2];
rxx(1.5716178) q[2],q[3];
rx(-0.85404295) q[0];
rx(-2.8680589) q[1];
rx(1.3015397) q[2];
rx(-2.0587289) q[3];
rz(0.37172952) q[0];
rz(2.4437354) q[1];
rz(1.722697) q[2];
rz(1.083079) q[3];
crx(3.383137) q[0],q[1];
crx(0.8861618) q[0],q[2];
crx(1.5916153) q[0],q[3];
crx(3.6753097) q[1],q[0];
crx(-2.9275677) q[1],q[2];
crx(-3.4011037) q[1],q[3];
crx(-0.83184689) q[2],q[0];
crx(3.8006427) q[2],q[1];
crx(-2.0805554) q[2],q[3];
crx(-1.5390357) q[3],q[0];
crx(-1.3912508) q[3],q[1];
crx(-2.8348203) q[3],q[2];
rx(-2.4439158) q[0];
rx(-0.98506296) q[1];
rx(-1.327455) q[2];
rx(2.0974433) q[3];
rz(0.2849561) q[0];
rz(-0.00043773942) q[1];
rz(0.15688029) q[2];
rz(0.025108727) q[3];
measure q[0] -> c[0];
measure q[1] -> c[1];
measure q[2] -> c[2];
measure q[3] -> c[3];