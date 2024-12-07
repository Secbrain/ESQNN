OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
creg c[4];
rz(0.036329839) q[0];
rz(2.4672697) q[1];
rz(-0.16547388) q[2];
rz(-0.30690777) q[3];
rx(1.4188533) q[0];
rx(-0.45662296) q[1];
rx(-1.5976079) q[2];
rx(0.77355069) q[3];
ry(-0.63600141) q[0];
ry(-0.2509535) q[1];
ry(0.70053798) q[2];
ry(1.4387873) q[3];
rz(-1.0684497) q[0];
rz(-0.16634242) q[1];
rz(0.51761156) q[2];
rz(-0.73252624) q[3];
rx(-1.0795935) q[0];
rx(2.9554987) q[1];
rx(2.9604213) q[2];
rx(2.989259) q[3];
rz(-2.3987236) q[0];
rz(0.92365491) q[1];
rz(-2.4862111) q[2];
rz(1.6308924) q[3];
crz(-1.5026062) q[0],q[1];
crz(-1.7930387) q[2],q[3];
rx(-0.89534873) q[0];
rx(1.4092419) q[1];
rx(-1.1747907) q[2];
rx(2.6897836) q[3];
rz(2.8086402) q[0];
rz(0.33810943) q[1];
rz(-2.6328323) q[2];
rz(0.75082141) q[3];
crz(-0.76248544) q[1],q[2];
rx(-2.8594637) q[0];
rx(-1.8577232) q[1];
rx(-2.454695) q[2];
rx(-1.1450907) q[3];
rz(-2.7577155) q[0];
rz(1.8392332) q[1];
rz(-1.5445412) q[2];
rz(0.33558601) q[3];
crz(-3.1127768) q[0],q[1];
crz(-1.3859648) q[2],q[3];
rx(0.57170188) q[0];
rx(1.8014185) q[1];
rx(1.0551311) q[2];
rx(-1.6450243) q[3];
rz(0.89805913) q[0];
rz(-1.5907699) q[1];
rz(-2.7450817) q[2];
rz(2.0019763) q[3];
crz(0.097339168) q[1],q[2];
rx(1.3429339) q[0];
rx(3.1460812) q[1];
rx(-0.042855013) q[2];
rx(1.5432197) q[3];
rz(0.75387681) q[0];
rz(-0.52791739) q[1];
rz(-2.6535525) q[2];
rz(0.50294149) q[3];
crz(-1.5050166) q[0],q[1];
crz(1.2890633) q[2],q[3];
rx(1.320684) q[0];
rx(-2.4921002) q[1];
rx(-1.4003389) q[2];
rx(1.8376302) q[3];
rz(-0.12833726) q[0];
rz(3.7973405e-07) q[1];
rz(0.78886998) q[2];
rz(-0.58808678) q[3];
crz(-0.0032447302) q[1],q[2];
measure q[0] -> c[0];
measure q[1] -> c[1];
measure q[2] -> c[2];
measure q[3] -> c[3];
