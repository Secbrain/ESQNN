OPENQASM 2.0;
include "qelib1.inc";
gate ryy(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(-0.26950851) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
gate ryy_139858775975488(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(-0.095033757) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
gate ryy_139858775975728(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(0.025443105) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
gate ryy_139858775974096(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(0.25280625) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
gate ryy_139858775974528(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(-0.34676963) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
gate ryy_139858775977456(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(-2.6788681) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
gate ryy_139858664129152(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(-0.5301227) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
gate ryy_139858664129872(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(2.4381697) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
gate ryy_139858664129056(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(0.77324867) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
gate ryy_139858664129584(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(0.32175118) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
gate ryy_139858664131408(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(-1.3879818) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
gate ryy_139858664129344(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(-1.9251845) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
gate ryy_139858662982656(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(-1.5900992) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
gate ryy_139858662982224(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(-1.2614565) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
gate ryy_139858662982512(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(0.95242667) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
gate ryy_139858663415712(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(0.056502022) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
gate ryy_139858663414272(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(0.054837469) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
gate ryy_139858663413120(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(0.10193484) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
gate ryy_139858663048432(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(-0.031170782) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
gate ryy_139858663048528(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(0.017428674) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
gate ryy_139858663048048(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(-0.061803952) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
gate ryy_139858663048768(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(-0.032447126) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
gate ryy_139858663048144(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(0.025802244) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
gate ryy_139858663049152(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(-1.0947129) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
qreg q[4];
creg c[4];
h q[0];
h q[1];
h q[2];
h q[3];
ry(-0.15868737) q[0];
ry(1.6983614) q[1];
ry(-0.055956144) q[2];
ry(-0.45469725) q[3];
ryy(-0.26950851) q[0],q[1];
ryy_139858775975488(-0.095033757) q[1],q[2];
ryy_139858775975728(0.025443105) q[2],q[3];
h q[0];
h q[1];
h q[2];
h q[3];
ry(1.2942348) q[0];
ry(0.1953326) q[1];
ry(-1.7752779) q[2];
ry(1.5089852) q[3];
ryy_139858775974096(0.25280625) q[0],q[1];
ryy_139858775974528(-0.34676963) q[1],q[2];
ryy_139858775977456(-2.6788681) q[2],q[3];
h q[0];
h q[1];
h q[2];
h q[3];
ry(-0.47618458) q[0];
ry(1.1132715) q[1];
ry(2.1900945) q[2];
ry(0.35306635) q[3];
ryy_139858664129152(-0.5301227) q[0],q[1];
ryy_139858664129872(2.4381697) q[1],q[2];
ryy_139858664129056(0.77324867) q[2],q[3];
h q[0];
h q[1];
h q[2];
h q[3];
ry(0.44739476) q[0];
ry(0.71916616) q[1];
ry(-1.9299877) q[2];
ry(0.99751127) q[3];
ryy_139858664129584(0.32175118) q[0],q[1];
ryy_139858664131408(-1.3879818) q[1],q[2];
ryy_139858664129344(-1.9251845) q[2],q[3];
h q[0];
h q[1];
h q[2];
h q[3];
ry(-1.3777537) q[0];
ry(1.1541244) q[1];
ry(-1.0929987) q[2];
ry(-0.87138861) q[3];
ryy_139858662982656(-1.5900992) q[0],q[1];
ryy_139858662982224(-1.2614565) q[1],q[2];
ryy_139858662982512(0.95242667) q[2],q[3];
h q[0];
h q[1];
h q[2];
h q[3];
ry(0.2234731) q[0];
ry(0.2528359) q[1];
ry(0.21688958) q[2];
ry(0.46998495) q[3];
ryy_139858663415712(0.056502022) q[0],q[1];
ryy_139858663414272(0.054837469) q[1],q[2];
ryy_139858663413120(0.10193484) q[2],q[3];
h q[0];
h q[1];
h q[2];
h q[3];
ry(0.32675457) q[0];
ry(-0.095395088) q[1];
ry(-0.1826999) q[2];
ry(0.33828124) q[3];
ryy_139858663048432(-0.031170782) q[0],q[1];
ryy_139858663048528(0.017428674) q[1],q[2];
ryy_139858663048048(-0.061803952) q[2],q[3];
h q[0];
h q[1];
h q[2];
h q[3];
ry(-0.69002211) q[0];
ry(0.047023315) q[1];
ry(0.54871172) q[2];
ry(-1.9950601) q[3];
ryy_139858663048768(-0.032447126) q[0],q[1];
ryy_139858663048144(0.025802244) q[1],q[2];
ryy_139858663049152(-1.0947129) q[2],q[3];
h q[0];
h q[1];
h q[2];
h q[3];
cz q[0],q[1];
cz q[1],q[2];
cz q[2],q[3];
rx(0.28306815) q[0];
rx(-2.784745) q[1];
rx(-2.1484423) q[2];
rx(-2.677633) q[3];
measure q[0] -> c[0];
measure q[1] -> c[1];
measure q[2] -> c[2];
measure q[3] -> c[3];
