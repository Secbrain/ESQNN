OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
creg c[4];
ry(-0.1591945) q[0];
ry(0.051534016) q[1];
ry(0.63793629) q[2];
ry(1.0802479) q[3];
rz(-1.2114757) q[0];
rz(0.60065365) q[1];
rz(0.076891094) q[2];
rz(-0.88476133) q[3];
rx(0.37095824) q[0];
rx(1.4747735) q[1];
rx(-1.7510506) q[2];
rx(-0.028895698) q[3];
ry(-0.66383529) q[0];
ry(-1.4844981) q[1];
ry(0.080916636) q[2];
ry(-0.54668027) q[3];
ry(-1.0514512) q[0];
ry(-0.64818072) q[1];
ry(-1.3570187) q[2];
ry(-1.3314624) q[3];
rz(0.51098585) q[0];
rz(-1.4619918) q[1];
rz(1.4522364) q[2];
rz(1.4651625) q[3];
rx(-0.1936288) q[0];
rx(-0.072854891) q[1];
rx(0.061060723) q[2];
rx(-0.61636686) q[3];
ry(1.0153071) q[0];
ry(0.59531993) q[1];
ry(0.76522923) q[2];
ry(1.9977002) q[3];
rx(-0.9760679) q[0];
rx(3.1815333) q[1];
rx(-1.8814567) q[2];
rx(-0.22039604) q[3];
rz(1.3306259e-10) q[0];
rz(4.1379414) q[1];
rz(2.2856631) q[2];
rz(1.1139243) q[3];
crx(-0.92064857) q[0],q[1];
crx(-0.53600055) q[1],q[2];
crx(-1.2657775) q[2],q[3];
measure q[0] -> c[0];
measure q[1] -> c[1];
measure q[2] -> c[2];
measure q[3] -> c[3];
