OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
creg c[4];
h q[0];
h q[1];
h q[2];
h q[3];
rz(1.0129451) q[0];
rz(-0.072181523) q[1];
rz(0.031169396) q[2];
rz(-0.7559849) q[3];
rzz(-0.073115915) q[0],q[1];
rzz(-0.0022498544) q[1],q[2];
rzz(-0.023563594) q[2],q[3];
h q[0];
h q[1];
h q[2];
h q[3];
rz(-0.79561907) q[0];
rz(-0.68681699) q[1];
rz(1.9091076) q[2];
rz(-0.49425983) q[3];
rzz(0.54644471) q[0],q[1];
rzz(-1.3112075) q[1],q[2];
rzz(-0.94359517) q[2],q[3];
h q[0];
h q[1];
h q[2];
h q[3];
rz(-0.063087903) q[0];
rz(0.95035899) q[1];
rz(0.66921753) q[2];
rz(0.12500714) q[3];
rzz(-0.059956156) q[0],q[1];
rzz(0.63599688) q[1],q[2];
rzz(0.083656967) q[2],q[3];
h q[0];
h q[1];
h q[2];
h q[3];
rz(0.50854623) q[0];
rz(-1.0646656) q[1];
rz(-0.46477252) q[2];
rz(-0.91526747) q[3];
rzz(-0.54143167) q[0],q[1];
rzz(0.4948273) q[1],q[2];
rzz(0.42539117) q[2],q[3];
ry(-1.779482) q[0];
ry(3.9671998) q[1];
ry(1.5464267) q[2];
ry(-3.4880581) q[3];
rz(2.0248933) q[0];
rz(-1.6403744) q[1];
rz(-1.8108302) q[2];
rz(3.0362024) q[3];
cx q[0],q[1];
cx q[2],q[3];
ry(0.011907264) q[1];
ry(-2.253165) q[2];
rz(0.27816853) q[1];
rz(0.62216032) q[2];
cx q[1],q[2];
ry(-0.64891505) q[0];
ry(1.7723136) q[1];
ry(3.1217048) q[2];
ry(-0.099556372) q[3];
rz(-2.8133318) q[0];
rz(-0.50055188) q[1];
rz(2.9043839) q[2];
rz(0.38648582) q[3];
cx q[0],q[1];
cx q[2],q[3];
ry(0.50014228) q[1];
ry(-1.6663412) q[2];
rz(2.1079469) q[1];
rz(1.6480349) q[2];
cx q[1],q[2];
ry(2.38745) q[0];
ry(1.8266011) q[1];
ry(1.3849015) q[2];
ry(3.0612543) q[3];
rz(-0.0054969504) q[0];
rz(1.3348091) q[1];
rz(0.46558622) q[2];
rz(-1.3682623) q[3];
cx q[0],q[1];
cx q[2],q[3];
ry(0.50425965) q[1];
ry(0.90621412) q[2];
rz(0.43135586) q[1];
rz(1.2524964e-10) q[2];
cx q[1],q[2];
measure q[0] -> c[0];
measure q[1] -> c[1];
measure q[2] -> c[2];
measure q[3] -> c[3];