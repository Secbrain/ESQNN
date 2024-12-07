OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
creg c[4];
h q[0];
h q[1];
h q[2];
h q[3];
rz(-0.17038141) q[0];
rz(-0.30279297) q[1];
rz(-1.2868071) q[2];
rz(-1.3662828) q[3];
rzz(0.051590294) q[0],q[1];
rzz(0.38963613) q[1],q[2];
rzz(1.7581424) q[2],q[3];
h q[0];
h q[1];
h q[2];
h q[3];
rz(-0.046252239) q[0];
rz(-0.61495847) q[1];
rz(1.2366945) q[2];
rz(-0.81435615) q[3];
rzz(0.028443206) q[0],q[1];
rzz(-0.76051575) q[1],q[2];
rzz(-1.0071098) q[2],q[3];
h q[0];
h q[1];
h q[2];
h q[3];
rz(1.14621) q[0];
rz(-1.1787332) q[1];
rz(-0.03667279) q[2];
rz(0.67180979) q[3];
rzz(-1.3510758) q[0],q[1];
rzz(0.043227434) q[1],q[2];
rzz(-0.024637138) q[2],q[3];
h q[0];
h q[1];
h q[2];
h q[3];
rz(0.92422974) q[0];
rz(0.2697157) q[1];
rz(0.62853712) q[2];
rz(-0.70661885) q[3];
rzz(0.24927928) q[0],q[1];
rzz(0.16952632) q[1],q[2];
rzz(-0.44413617) q[2],q[3];
ry(1.4159254) q[0];
ry(1.04356) q[1];
ry(0.47657016) q[2];
ry(2.9116135) q[3];
rz(-0.22423983) q[0];
rz(0.053912319) q[1];
rz(-1.0068709) q[2];
rz(-2.1047257e-09) q[3];
cz q[0],q[1];
cz q[2],q[3];
ry(-0.83187491) q[1];
ry(1.1911469) q[2];
rz(-2.2831406e-10) q[1];
rz(-0.808761) q[2];
cz q[1],q[2];
measure q[0] -> c[0];
measure q[1] -> c[1];
measure q[2] -> c[2];
measure q[3] -> c[3];
