OPENQASM 2.0;
include "qelib1.inc";
gate ryy(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(2.8666337) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
gate ryy_139858663415712(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(-0.19909973) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
gate ryy_139858663412544(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(-0.035000972) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
gate ryy_139858664155456(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(0.070982903) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
gate ryy_139858664153200(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(0.054079656) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
gate ryy_139858664154400(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(0.91354501) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
gate ryy_139858765148416(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(0.85129988) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
gate ryy_139858765151056(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(0.99757862) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
gate ryy_139858664049008(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(-1.097315) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
gate ryy_139858664138688(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(-1.3572019) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
gate ryy_139858664137968(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(3.179317) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
gate ryy_139858663812928(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(-1.45516) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
gate ryy_139858775925520(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(0.031612858) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
gate ryy_139858663472048(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(0.63550431) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
gate ryy_139858663471856(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(-0.053000771) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
gate ryy_139858765117712(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(-0.25646153) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
gate ryy_139858765116224(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(-0.55022979) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
gate ryy_139858765119200(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(0.96899408) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
gate ryy_139858775964352(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(-0.90016794) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
gate ryy_139858775961664(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(0.38959181) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
gate ryy_139858775962480(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(0.24579681) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
gate ryy_139858662641184(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(-0.38859236) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
gate ryy_139858662640176(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(0.48708919) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
gate ryy_139858662640080(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(0.48179427) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
qreg q[4];
creg c[4];
h q[0];
h q[1];
h q[2];
h q[3];
ry(-1.4647197) q[0];
ry(-1.957121) q[1];
ry(0.10173093) q[2];
ry(-0.3440544) q[3];
ryy(2.8666337) q[0],q[1];
ryy_139858663415712(-0.19909973) q[1],q[2];
ryy_139858663412544(-0.035000972) q[2],q[3];
h q[0];
h q[1];
h q[2];
h q[3];
ry(1.2444955) q[0];
ry(0.057037495) q[1];
ry(0.94814217) q[2];
ry(0.96351057) q[3];
ryy_139858664155456(0.070982903) q[0],q[1];
ryy_139858664153200(0.054079656) q[1],q[2];
ryy_139858664154400(0.91354501) q[2],q[3];
h q[0];
h q[1];
h q[2];
h q[3];
ry(0.69314545) q[0];
ry(1.2281692) q[1];
ry(0.81224853) q[2];
ry(-1.3509595) q[3];
ryy_139858765148416(0.85129988) q[0],q[1];
ryy_139858765151056(0.99757862) q[1],q[2];
ryy_139858664049008(-1.097315) q[2],q[3];
h q[0];
h q[1];
h q[2];
h q[3];
ry(-0.62316936) q[0];
ry(2.1779022) q[1];
ry(1.459807) q[2];
ry(-0.99681669) q[3];
ryy_139858664138688(-1.3572019) q[0],q[1];
ryy_139858664137968(3.179317) q[1],q[2];
ryy_139858663812928(-1.45516) q[2],q[3];
h q[0];
h q[1];
h q[2];
h q[3];
ry(-0.08842548) q[0];
ry(-0.35750848) q[1];
ry(-1.7775923) q[2];
ry(0.029816044) q[3];
ryy_139858775925520(0.031612858) q[0],q[1];
ryy_139858663472048(0.63550431) q[1],q[2];
ryy_139858663471856(-0.053000771) q[2],q[3];
h q[0];
h q[1];
h q[2];
h q[3];
ry(0.36385661) q[0];
ry(-0.70484233) q[1];
ry(0.78064233) q[2];
ry(1.2412779) q[3];
ryy_139858765117712(-0.25646153) q[0],q[1];
ryy_139858765116224(-0.55022979) q[1],q[2];
ryy_139858765119200(0.96899408) q[2],q[3];
h q[0];
h q[1];
h q[2];
h q[3];
ry(-1.2911437) q[0];
ry(0.69718653) q[1];
ry(0.5588057) q[2];
ry(0.43986097) q[3];
ryy_139858775964352(-0.90016794) q[0],q[1];
ryy_139858775961664(0.38959181) q[1],q[2];
ryy_139858775962480(0.24579681) q[2],q[3];
h q[0];
h q[1];
h q[2];
h q[3];
ry(0.67420971) q[0];
ry(-0.5763672) q[1];
ry(-0.84510219) q[2];
ry(-0.5701018) q[3];
ryy_139858662641184(-0.38859236) q[0],q[1];
ryy_139858662640176(0.48708919) q[1],q[2];
ryy_139858662640080(0.48179427) q[2],q[3];
rx(-0.9368574) q[0];
rx(-2.3727372) q[1];
rx(3.0943372) q[2];
rx(0.52820206) q[3];
rz(-0.009535036) q[0];
rz(-0.24169451) q[1];
rz(0.00034412532) q[2];
rz(-6.8425538e-06) q[3];
crz(3.6798065e-07) q[0],q[1];
crz(-1.263358e-09) q[2],q[3];
crz(0.38128471) q[1],q[2];
measure q[0] -> c[0];
measure q[1] -> c[1];
measure q[2] -> c[2];
measure q[3] -> c[3];