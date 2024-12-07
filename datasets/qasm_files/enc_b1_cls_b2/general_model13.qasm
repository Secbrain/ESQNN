OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
creg c[4];
ry(-0.29195806) q[0];
ry(0.45643651) q[1];
ry(-0.31470188) q[2];
ry(-0.41328999) q[3];
rz(0.3946206) q[0];
rz(1.1304747) q[1];
rz(0.82583827) q[2];
rz(0.94582766) q[3];
rx(-0.15447335) q[0];
rx(-1.6013477) q[1];
rx(-0.059470855) q[2];
rx(-0.99286664) q[3];
ry(1.1634254) q[0];
ry(1.6094594) q[1];
ry(-0.29416555) q[2];
ry(1.081931) q[3];
ry(3.3807263) q[0];
ry(1.5521395) q[1];
ry(0.58826113) q[2];
ry(-2.311965) q[3];
crz(0.42477617) q[0],q[3];
crz(-1.1860858) q[1],q[0];
crz(-1.0753044) q[2],q[1];
crz(1.2042129) q[3],q[2];
ry(2.2076325) q[0];
ry(1.3301456) q[1];
ry(-0.85737437) q[2];
ry(1.7893404) q[3];
crz(-1.7003443) q[0],q[1];
crz(2.7157295) q[3],q[0];
crz(-2.089653) q[2],q[3];
crz(1.3393081) q[1],q[2];
ry(0.81217855) q[0];
ry(-3.2322989) q[1];
ry(1.9636399) q[2];
ry(0.16864394) q[3];
crz(-2.925333) q[0],q[3];
crz(0.72436732) q[1],q[0];
crz(-0.26048908) q[2],q[1];
crz(-0.90324646) q[3],q[2];
ry(1.009076) q[0];
ry(0.47863603) q[1];
ry(1.1486537) q[2];
ry(-2.5859704) q[3];
crz(-0.67874467) q[0],q[1];
crz(-0.0037899718) q[3],q[0];
crz(-3.345664e-10) q[2],q[3];
crz(-0.61864203) q[1],q[2];
measure q[0] -> c[0];
measure q[1] -> c[1];
measure q[2] -> c[2];
measure q[3] -> c[3];