OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
creg c[4];
h q[0];
h q[1];
h q[2];
h q[3];
rz(-0.9080928) q[0];
rz(1.1592088) q[1];
rz(-1.4673603) q[2];
rz(1.7191014) q[3];
rzz(-1.0526692) q[0],q[1];
rzz(-1.7009768) q[1],q[2];
rzz(-2.522541) q[2],q[3];
h q[0];
h q[1];
h q[2];
h q[3];
rz(-0.36731902) q[0];
rz(2.357583) q[1];
rz(-0.4597221) q[2];
rz(-0.058640674) q[3];
rzz(-0.8659851) q[0],q[1];
rzz(-1.083833) q[1],q[2];
rzz(0.026958413) q[2],q[3];
h q[0];
h q[1];
h q[2];
h q[3];
rz(-1.2418878) q[0];
rz(-0.023634955) q[1];
rz(-2.0428922) q[2];
rz(0.19053656) q[3];
rzz(0.029351963) q[0],q[1];
rzz(0.048283666) q[1],q[2];
rzz(-0.38924566) q[2],q[3];
h q[0];
h q[1];
h q[2];
h q[3];
rz(0.57569927) q[0];
rz(-0.51776165) q[1];
rz(0.91970748) q[2];
rz(0.98618215) q[3];
rzz(-0.29807499) q[0],q[1];
rzz(-0.47618926) q[1],q[2];
rzz(0.90699911) q[2],q[3];
rx(0.19415632) q[0];
rx(1.7239037) q[1];
rx(-2.1837883) q[2];
rx(1.3544985) q[3];
rz(1.2150716) q[0];
rz(0.62172806) q[1];
rz(-0.63593745) q[2];
rz(-1.2622961) q[3];
crz(-1.3597411) q[0],q[1];
crz(0.84031397) q[1],q[2];
crz(0.29777509) q[2],q[3];
rx(1.2098747) q[0];
rx(1.4837617) q[1];
rx(-2.5584271) q[2];
rx(2.8954873) q[3];
rz(2.1227312) q[0];
rz(-3.9451938) q[1];
rz(-3.5435324) q[2];
rz(2.1960611) q[3];
crz(-2.2783573) q[0],q[1];
crz(1.6825171) q[1],q[2];
crz(3.2062876) q[2],q[3];
rx(-1.7916456) q[0];
rx(-2.4416292) q[1];
rx(-0.48369297) q[2];
rx(-1.1364315) q[3];
rz(1.0671419e-09) q[0];
rz(-9.4004513e-08) q[1];
rz(0.94063008) q[2];
rz(-0.20948321) q[3];
crz(1.4506321e-11) q[0],q[1];
crz(0.097169302) q[1],q[2];
crz(-0.022975331) q[2],q[3];
measure q[0] -> c[0];
measure q[1] -> c[1];
measure q[2] -> c[2];
measure q[3] -> c[3];
