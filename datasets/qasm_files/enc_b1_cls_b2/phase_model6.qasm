OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
creg c[4];
rx(-1.9005532) q[0];
rx(0.22857653) q[1];
rx(0.024859404) q[2];
rx(-0.34595025) q[3];
rx(0.28683281) q[0];
rx(-0.73084241) q[1];
rx(0.17482026) q[2];
rx(-1.0939293) q[3];
rx(-1.6021603) q[0];
rx(1.3528969) q[1];
rx(1.2888277) q[2];
rx(0.052295472) q[3];
rx(-1.5468504) q[0];
rx(0.75670606) q[1];
rx(0.77551949) q[2];
rx(2.0265355) q[3];
rx(1.2047641) q[0];
rx(3.0185547) q[1];
rx(1.2050241) q[2];
rx(1.1772083) q[3];
rz(-0.090019174) q[0];
rz(1.7603209) q[1];
rz(0.65095562) q[2];
rz(-1.9415425) q[3];
crx(1.2591958) q[0],q[1];
crx(-0.16372544) q[0],q[2];
crx(0.2845324) q[0],q[3];
crx(-2.2942913) q[1],q[0];
crx(-0.34442624) q[1],q[2];
crx(-2.4111686) q[1],q[3];
crx(-1.8688244) q[2],q[0];
crx(3.0108178) q[2],q[1];
crx(-2.0824676) q[2],q[3];
crx(-0.48849037) q[3],q[0];
crx(0.22418483) q[3],q[1];
crx(0.65490967) q[3],q[2];
rx(-0.56185019) q[0];
rx(-1.4492763) q[1];
rx(-1.9835007) q[2];
rx(0.39587852) q[3];
rz(-3.4433017) q[0];
rz(-2.98156) q[1];
rz(3.0772445) q[2];
rz(2.0504093) q[3];
rx(-1.4459937) q[0];
rx(-1.7596587) q[1];
rx(2.1342342) q[2];
rx(-0.1235163) q[3];
rz(-2.433496) q[0];
rz(0.99438822) q[1];
rz(2.2056983) q[2];
rz(1.3861207) q[3];
crx(3.1077271) q[0],q[1];
crx(-2.2358563) q[0],q[2];
crx(-0.73477805) q[0],q[3];
crx(5.2113252) q[1],q[0];
crx(-2.9689758) q[1],q[2];
crx(-1.5800114) q[1],q[3];
crx(-0.72369695) q[2],q[0];
crx(0.18718761) q[2],q[1];
crx(-0.072083175) q[2],q[3];
crx(1.6792121) q[3],q[0];
crx(-1.0621058) q[3],q[1];
crx(1.1137133) q[3],q[2];
rx(-3.2480056) q[0];
rx(-2.755363) q[1];
rx(2.2292559) q[2];
rx(-2.1755483) q[3];
rz(-1.3854466e-09) q[0];
rz(0.0036346314) q[1];
rz(4.9144916e-07) q[2];
rz(-0.074420609) q[3];
measure q[0] -> c[0];
measure q[1] -> c[1];
measure q[2] -> c[2];
measure q[3] -> c[3];
