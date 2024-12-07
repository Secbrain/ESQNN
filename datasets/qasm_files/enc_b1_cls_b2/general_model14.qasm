OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
creg c[4];
ry(-0.93805116) q[0];
ry(-0.63850629) q[1];
ry(0.21946865) q[2];
ry(-0.43924296) q[3];
rz(-0.13910709) q[0];
rz(-0.018692156) q[1];
rz(1.6560721) q[2];
rz(1.0661179) q[3];
rx(-4/(7*pi)) q[0];
rx(-1.2379671) q[1];
rx(0.51422012) q[2];
rx(-0.15104349) q[3];
ry(0.13773604) q[0];
ry(1.2250829) q[1];
ry(-0.76426029) q[2];
ry(0.91838348) q[3];
ry(-3.121743) q[0];
ry(-0.021951059) q[1];
ry(-2.8948352) q[2];
ry(3.6454737) q[3];
crx(-0.11551467) q[0],q[3];
crx(-1.0522333) q[1],q[0];
crx(-0.60503894) q[2],q[1];
crx(2.9685545) q[3],q[2];
ry(1.8582852) q[0];
ry(2.1584489) q[1];
ry(-0.17808989) q[2];
ry(0.32961291) q[3];
crx(-1.8387132) q[0],q[1];
crx(1.1951706) q[3],q[0];
crx(1.2093246) q[2],q[3];
crx(1.4428872) q[1],q[2];
ry(2.4951861) q[0];
ry(-2.8545458) q[1];
ry(-1.0007204) q[2];
ry(1.6720303) q[3];
crx(-0.08525838) q[0],q[3];
crx(-1.557889) q[1],q[0];
crx(0.9341929) q[2],q[1];
crx(-0.77998465) q[3],q[2];
ry(-0.61156082) q[0];
ry(0.032169592) q[1];
ry(-1.626812) q[2];
ry(1.4267359) q[3];
crx(-0.22282648) q[0],q[1];
crx(4.4998512) q[3],q[0];
crx(2.9894083) q[2],q[3];
crx(-1.6842918) q[1],q[2];
measure q[0] -> c[0];
measure q[1] -> c[1];
measure q[2] -> c[2];
measure q[3] -> c[3];
