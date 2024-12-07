OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
creg c[4];
rx(-0.44770017) q[0];
rx(-0.72881693) q[1];
rx(-0.16066237) q[2];
rx(-0.32063714) q[3];
ry(-0.63077378) q[0];
ry(-0.78876662) q[1];
ry(1.3061576) q[2];
ry(-0.92757636) q[3];
rz(-0.26273948) q[0];
rz(0.93149525) q[1];
rz(-0.45934671) q[2];
rz(-0.94194567) q[3];
rx(-0.70891863) q[0];
rx(2.1860759) q[1];
rx(-0.64931589) q[2];
rx(0.45214036) q[3];
ry(-0.69812465) q[0];
ry(3.6931579) q[1];
ry(0.27028054) q[2];
ry(-3.8765581) q[3];
rz(0.0060184388) q[0];
rz(0.43433458) q[1];
rz(-0.66061229) q[2];
rz(-0.15282488) q[3];
cz q[0],q[1];
cz q[2],q[3];
ry(1.0682409) q[1];
ry(-2.1876392) q[2];
rz(0.74570906) q[1];
rz(-0.43918809) q[2];
cz q[1],q[2];
measure q[0] -> c[0];
measure q[1] -> c[1];
measure q[2] -> c[2];
measure q[3] -> c[3];
