OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
creg c[4];
h q[0];
h q[1];
h q[2];
h q[3];
rz(-0.93805116) q[0];
rz(-0.63850629) q[1];
rz(0.21946865) q[2];
rz(-0.43924296) q[3];
rzz(0.59895158) q[0],q[1];
rzz(-0.14013211) q[1],q[2];
rzz(-0.09640006) q[2],q[3];
h q[0];
h q[1];
h q[2];
h q[3];
rz(-0.13910709) q[0];
rz(-0.018692156) q[1];
rz(1.6560721) q[2];
rz(1.0661179) q[3];
rzz(0.0026002114) q[0],q[1];
rzz(-0.030955559) q[1],q[2];
rzz(1.7655681) q[2],q[3];
h q[0];
h q[1];
h q[2];
h q[3];
rz(-4/(7*pi)) q[0];
rz(-1.2379671) q[1];
rz(0.51422012) q[2];
rz(-0.15104349) q[3];
rzz(0.22517619) q[0],q[1];
rzz(-0.63658762) q[1],q[2];
rzz(-0.077669598) q[2],q[3];
h q[0];
h q[1];
h q[2];
h q[3];
rz(0.13773604) q[0];
rz(1.2250829) q[1];
rz(-0.76426029) q[2];
rz(0.91838348) q[3];
rzz(0.16873807) q[0],q[1];
rzz(-0.93628222) q[1],q[2];
rzz(-0.70188403) q[2],q[3];
ry(-2.8446417) q[0];
ry(-1.6087197) q[1];
ry(-3.6408238) q[2];
ry(-0.99557143) q[3];
crx(-0.88225186) q[0],q[3];
crx(-1.9479324) q[1],q[0];
crx(1.2260904) q[2],q[1];
crx(3.580617) q[3],q[2];
ry(-1.4027798) q[0];
ry(2.9108973) q[1];
ry(-0.59061086) q[2];
ry(1.9730827) q[3];
crx(1.2572944) q[0],q[1];
crx(-1.053454) q[3],q[0];
crx(-2.0808508) q[2],q[3];
crx(2.1425366) q[1],q[2];
measure q[0] -> c[0];
measure q[1] -> c[1];
measure q[2] -> c[2];
measure q[3] -> c[3];
