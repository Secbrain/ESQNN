OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
creg c[4];
ry(-1.8807623) q[0];
ry(1.4751683) q[1];
ry(1.6362823) q[2];
ry(-0.96454078) q[3];
rz(1.140761) q[0];
rz(0.41566601) q[1];
rz(2.0262315) q[2];
rz(-1.0126259) q[3];
rx(0.34856999) q[0];
rx(0.58184922) q[1];
rx(-0.3934761) q[2];
rx(0.45355415) q[3];
ry(-1.1785884) q[0];
ry(0.78981906) q[1];
ry(1.1420684) q[2];
ry(0.5569579) q[3];
ry(0.12128927) q[0];
ry(0.44558772) q[1];
ry(-0.97702706) q[2];
ry(-0.58459711) q[3];
rz(-1.5499167) q[0];
rz(0.30215302) q[1];
rz(-0.34727851) q[2];
rz(-0.2026305) q[3];
rx(-0.44323164) q[0];
rx(1.2706386) q[1];
rx(-0.67737305) q[2];
rx(0.647861) q[3];
ry(-0.97558272) q[0];
ry(1.8391463) q[1];
ry(-0.003486508) q[2];
ry(0.41038656) q[3];
rx(-1.2166282) q[0];
rx(-2.7208571) q[1];
rx(4.5619502) q[2];
rx(-0.48348972) q[3];
rz(-7.410784e-07) q[0];
rz(1.2155785e-09) q[1];
rz(-0.71899295) q[2];
rz(0.9050442) q[3];
measure q[0] -> c[0];
measure q[1] -> c[1];
measure q[2] -> c[2];
measure q[3] -> c[3];
