OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
creg c[4];
h q[0];
h q[1];
h q[2];
h q[3];
rx(-1.5537862) q[0];
rx(-0.90002882) q[1];
rx(1.1800689) q[2];
rx(-1.0849788) q[3];
rxx(1.3984523) q[0],q[1];
rxx(-1.062096) q[1],q[2];
rxx(-1.2803497) q[2],q[3];
h q[0];
h q[1];
h q[2];
h q[3];
rx(0.66620153) q[0];
rx(-0.72531396) q[1];
rx(1.3528261) q[2];
rx(-0.13646416) q[3];
rxx(-0.48320526) q[0],q[1];
rxx(-0.98122364) q[1],q[2];
rxx(-0.18461229) q[2],q[3];
h q[0];
h q[1];
h q[2];
h q[3];
rx(1.1062331) q[0];
rx(-0.2775273) q[1];
rx(-0.73012698) q[2];
rx(-0.90588683) q[3];
rxx(-0.30700991) q[0],q[1];
rxx(0.20263018) q[1],q[2];
rxx(0.66141242) q[2],q[3];
h q[0];
h q[1];
h q[2];
h q[3];
rx(0.33955121) q[0];
rx(0.95408165) q[1];
rx(-0.042208631) q[2];
rx(0.059119936) q[3];
rxx(0.32395959) q[0],q[1];
rxx(-0.040270481) q[1],q[2];
rxx(-0.0024953715) q[2],q[3];
h q[0];
h q[1];
h q[2];
h q[3];
rx(-0.04415971) q[0];
rx(0.65738744) q[1];
rx(-1.8288997) q[2];
rx(0.51574653) q[3];
rxx(-0.029030038) q[0],q[1];
rxx(-1.2022957) q[1],q[2];
rxx(-0.94324869) q[2],q[3];
h q[0];
h q[1];
h q[2];
h q[3];
rx(-2.181325) q[0];
rx(-0.74432933) q[1];
rx(0.46014088) q[2];
rx(1.2225194) q[3];
rxx(1.6236242) q[0],q[1];
rxx(-0.34249637) q[1],q[2];
rxx(0.56253117) q[2],q[3];
h q[0];
h q[1];
h q[2];
h q[3];
rx(-2.0475724) q[0];
rx(1.0944718) q[1];
rx(0.42030036) q[2];
rx(-0.63664818) q[3];
rxx(-2.2410102) q[0],q[1];
rxx(0.46000689) q[1],q[2];
rxx(-0.26758346) q[2],q[3];
h q[0];
h q[1];
h q[2];
h q[3];
rx(-2.1695642) q[0];
rx(1.1429639) q[1];
rx(0.89557397) q[2];
rx(1.6797636) q[3];
rxx(-2.4797335) q[0],q[1];
rxx(1.0236087) q[1],q[2];
rxx(1.5043526) q[2],q[3];
ry(-0.41240916) q[0];
ry(-0.95967907) q[1];
ry(3.1828058) q[2];
ry(-0.30864123) q[3];
rz(-0.033478666) q[0];
rz(0.28716987) q[1];
rz(-0.30120584) q[2];
rz(-0.23626713) q[3];
cx q[0],q[1];
cx q[2],q[3];
ry(1.1870257) q[1];
ry(-1.490388) q[2];
rz(0.0059322338) q[1];
rz(-3.4644099e-10) q[2];
cx q[1],q[2];
measure q[0] -> c[0];
measure q[1] -> c[1];
measure q[2] -> c[2];
measure q[3] -> c[3];