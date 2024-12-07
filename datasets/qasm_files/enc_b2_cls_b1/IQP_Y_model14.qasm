OPENQASM 2.0;
include "qelib1.inc";
gate ryy(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(0.26123849) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
gate ryy_139858765178816(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(-0.057603862) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
gate ryy_139858765179296(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(-0.35425732) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
gate ryy_139858662947664(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(0.68371576) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
gate ryy_139858765309360(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(0.58362538) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
gate ryy_139858765310224(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(-0.14860255) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
gate ryy_139858664033632(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(0.063790075) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
gate ryy_139858664033536(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(2.3413568) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
gate ryy_139858664030416(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(0.56589735) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
gate ryy_139858664052096(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(-0.15955026) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
gate ryy_139858664139216(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(-1.6050541) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
gate ryy_139858664136816(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(-2.714427) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
gate ryy_139858663093296(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(0.10876882) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
gate ryy_139858663093056(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(-0.26208925) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
gate ryy_139858663094496(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(0.055593375) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
gate ryy_139858663092960(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(-0.018225441) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
gate ryy_139858663095888(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(-0.19551976) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
gate ryy_139858663092816(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(0.21519606) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
gate ryy_139858663470320(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(0.60728848) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
gate ryy_139858663471664(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(1.0691491) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
gate ryy_139858663470896(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(0.244381) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
gate ryy_139858663425936(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(1.5288454) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
gate ryy_139858663427472(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(0.35430768) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
gate ryy_139858663427328(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(0.32608837) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
qreg q[4];
creg c[4];
h q[0];
h q[1];
h q[2];
h q[3];
ry(-1.3241873) q[0];
ry(-0.19728212) q[1];
ry(0.29198724) q[2];
ry(-1.213263) q[3];
ryy(0.26123849) q[0],q[1];
ryy_139858765178816(-0.057603862) q[1],q[2];
ryy_139858765179296(-0.35425732) q[2],q[3];
h q[0];
h q[1];
h q[2];
h q[3];
ry(0.73706281) q[0];
ry(0.92762208) q[1];
ry(0.62916291) q[2];
ry(-0.23619089) q[3];
ryy_139858662947664(0.68371576) q[0],q[1];
ryy_139858765309360(0.58362538) q[1],q[2];
ryy_139858765310224(-0.14860255) q[2],q[3];
h q[0];
h q[1];
h q[2];
h q[3];
ry(-0.022938488) q[0];
ry(-2.7809188) q[1];
ry(-0.84193641) q[2];
ry(-0.67213786) q[3];
ryy_139858664033632(0.063790075) q[0],q[1];
ryy_139858664033536(2.3413568) q[1],q[2];
ryy_139858664030416(0.56589735) q[2],q[3];
h q[0];
h q[1];
h q[2];
h q[3];
ry(0.13255943) q[0];
ry(-1.203613) q[1];
ry(1.3335301) q[2];
ry(-2.0355198) q[3];
ryy_139858664052096(-0.15955026) q[0],q[1];
ryy_139858664139216(-1.6050541) q[1],q[2];
ryy_139858664136816(-2.714427) q[2],q[3];
h q[0];
h q[1];
h q[2];
h q[3];
ry(0.2778891) q[0];
ry(0.39141089) q[1];
ry(-0.66960132) q[2];
ry(-0.083024591) q[3];
ryy_139858663093296(0.10876882) q[0],q[1];
ryy_139858663093056(-0.26208925) q[1],q[2];
ryy_139858663094496(0.055593375) q[2],q[3];
h q[0];
h q[1];
h q[2];
h q[3];
ry(0.069178484) q[0];
ry(-0.26345533) q[1];
ry(0.7421363) q[2];
ry(0.28996837) q[3];
ryy_139858663092960(-0.018225441) q[0],q[1];
ryy_139858663095888(-0.19551976) q[1],q[2];
ryy_139858663092816(0.21519606) q[2],q[3];
h q[0];
h q[1];
h q[2];
h q[3];
ry(-0.43808052) q[0];
ry(-1.3862486) q[1];
ry(-0.77125359) q[2];
ry(-0.31686205) q[3];
ryy_139858663470320(0.60728848) q[0],q[1];
ryy_139858663471664(1.0691491) q[1],q[2];
ryy_139858663470896(0.244381) q[2],q[3];
h q[0];
h q[1];
h q[2];
h q[3];
ry(-1.6128871) q[0];
ry(-0.94789362) q[1];
ry(-0.37378421) q[2];
ry(-0.87239736) q[3];
ryy_139858663425936(1.5288454) q[0],q[1];
ryy_139858663427472(0.35430768) q[1],q[2];
ryy_139858663427328(0.32608837) q[2],q[3];
ry(-2.106071) q[0];
ry(-1.5063012) q[1];
ry(-2.7290611) q[2];
ry(-1.7018013) q[3];
crx(-0.56009138) q[0],q[3];
crx(1.1797832) q[1],q[0];
crx(0.55381453) q[2],q[1];
crx(1.1918437) q[3],q[2];
ry(-1.1426816) q[0];
ry(2.2010448) q[1];
ry(-0.83866507) q[2];
ry(1.0199434) q[3];
crx(0.79552579) q[0],q[1];
crx(-1.7255607) q[3],q[0];
crx(-3.3097517) q[2],q[3];
crx(0.42064708) q[1],q[2];
measure q[0] -> c[0];
measure q[1] -> c[1];
measure q[2] -> c[2];
measure q[3] -> c[3];