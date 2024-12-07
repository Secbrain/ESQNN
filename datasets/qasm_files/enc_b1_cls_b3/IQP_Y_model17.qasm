OPENQASM 2.0;
include "qelib1.inc";
gate ryy(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(2.2228179) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
gate ryy_140342689780352(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(-1.5243558) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
gate ryy_140342689780880(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(0.47370824) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
gate ryy_140342723802160(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(1.0333426) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
gate ryy_140342689748016(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(0.67391741) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
gate ryy_140342689748736(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(-0.62682182) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
gate ryy_140342689999312(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(-0.24448684) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
gate ryy_140342690394064(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(0.10127584) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
gate ryy_140342690390368(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(0.18494563) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
gate ryy_140342690391328(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(0.0365397) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
gate ryy_140342690391040(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(3.426486) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
gate ryy_140342690390272(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(-0.80741268) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
qreg q[4];
creg c[4];
h q[0];
h q[1];
h q[2];
h q[3];
ry(2.0818412) q[0];
ry(1.0677173) q[1];
ry(-1.4276773) q[2];
ry(-0.33180344) q[3];
ryy(2.2228179) q[0],q[1];
ryy_140342689780352(-1.5243558) q[1],q[2];
ryy_140342689780880(0.47370824) q[2],q[3];
h q[0];
h q[1];
h q[2];
h q[3];
ry(1.7054012) q[0];
ry(0.60592347) q[1];
ry(1.1122154) q[2];
ry(-0.5635795) q[3];
ryy_140342723802160(1.0333426) q[0],q[1];
ryy_140342689748016(0.67391741) q[1],q[2];
ryy_140342689748736(-0.62682182) q[2],q[3];
h q[0];
h q[1];
h q[2];
h q[3];
ry(-1.3645277) q[0];
ry(0.17917323) q[1];
ry(0.56523979) q[2];
ry(0.32719854) q[3];
ryy_140342689999312(-0.24448684) q[0],q[1];
ryy_140342690394064(0.10127584) q[1],q[2];
ryy_140342690390368(0.18494563) q[2],q[3];
h q[0];
h q[1];
h q[2];
h q[3];
ry(0.013574255) q[0];
ry(2.6918385) q[1];
ry(1.2729167) q[2];
ry(-0.6343013) q[3];
ryy_140342690391328(0.0365397) q[0],q[1];
ryy_140342690391040(3.426486) q[1],q[2];
ryy_140342690390272(-0.80741268) q[2],q[3];
rx(-2.270905) q[0];
rx(-1.8424404) q[1];
rx(1.1571673) q[2];
rx(1.151764) q[3];
rz(0.27973771) q[0];
rz(2.0682695) q[1];
rz(1.1021585) q[2];
rz(1.0603573) q[3];
crz(0.69426668) q[0],q[1];
crz(0.067120552) q[2],q[3];
rx(2.1368732) q[0];
rx(2.8833883) q[1];
rx(1.2219274) q[2];
rx(0.908095) q[3];
rz(-0.33828089) q[0];
rz(2.2921646) q[1];
rz(1.9973739) q[2];
rz(2.4260705) q[3];
crz(-1.1581602) q[1],q[2];
rx(0.14448327) q[0];
rx(-2.7286146) q[1];
rx(-3.1667857) q[2];
rx(2.8112755) q[3];
rz(-1.1506566) q[0];
rz(-2.3271916) q[1];
rz(-1.5666604) q[2];
rz(1.7603315) q[3];
crz(-2.7532892) q[0],q[1];
crz(-4.3983316) q[2],q[3];
rx(1.1029985) q[0];
rx(-1.3416065) q[1];
rx(-1.98754) q[2];
rx(-0.086402588) q[3];
rz(-0.98092592) q[0];
rz(-0.69468325) q[1];
rz(-1.7879242) q[2];
rz(-0.78718269) q[3];
crz(2.8372588) q[1],q[2];
rx(-0.49975786) q[0];
rx(-2.2210851) q[1];
rx(-1.2449839) q[2];
rx(-1.9150401) q[3];
rz(2.0155129) q[0];
rz(1.5712837) q[1];
rz(2.8951674) q[2];
rz(1.0537513) q[3];
crz(-0.5676465) q[0],q[1];
crz(2.8808589) q[2],q[3];
rx(-1.3193644) q[0];
rx(1.9661773) q[1];
rx(1.1357896) q[2];
rx(2.6962173) q[3];
rz(0.51954561) q[0];
rz(-0.0012493748) q[1];
rz(-0.0044073137) q[2];
rz(-3.1111011e-07) q[3];
crz(3.0665584e-10) q[1],q[2];
measure q[0] -> c[0];
measure q[1] -> c[1];
measure q[2] -> c[2];
measure q[3] -> c[3];
