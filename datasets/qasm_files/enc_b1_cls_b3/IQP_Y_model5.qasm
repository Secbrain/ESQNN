OPENQASM 2.0;
include "qelib1.inc";
gate ryy(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(-0.83399713) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
gate ryy_140342723854592(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(-0.16988647) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
gate ryy_140342723854496(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(-0.10383571) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
gate ryy_140342723804368(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(1.1204517) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
gate ryy_140342723802160(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(-1.2328316) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
gate ryy_140342723802400(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(0.30680659) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
gate ryy_140342723849568(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(-3.0692952) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
gate ryy_140342723849664(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(2.681782) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
gate ryy_140342723846976(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(-0.68840069) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
gate ryy_140342723819984(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(0.22002716) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
gate ryy_140342723817728(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(-0.56065106) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
gate ryy_140342723818400(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(0.96522439) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
qreg q[4];
creg c[4];
h q[0];
h q[1];
h q[2];
h q[3];
ry(1.1232847) q[0];
ry(-0.74246281) q[1];
ry(0.2288148) q[2];
ry(-0.45379806) q[3];
ryy(-0.83399713) q[0],q[1];
ryy_140342723854592(-0.16988647) q[1],q[2];
ryy_140342723854496(-0.10383571) q[2],q[3];
h q[0];
h q[1];
h q[2];
h q[3];
ry(0.79739141) q[0];
ry(1.4051464) q[1];
ry(-0.87736881) q[2];
ry(-0.34968942) q[3];
ryy_140342723804368(1.1204517) q[0],q[1];
ryy_140342723802160(-1.2328316) q[1],q[2];
ryy_140342723802400(0.30680659) q[2],q[3];
h q[0];
h q[1];
h q[2];
h q[3];
ry(-2.4602458) q[0];
ry(1.2475563) q[1];
ry(2.1496279) q[2];
ry(-0.32024178) q[3];
ryy_140342723849568(-3.0692952) q[0],q[1];
ryy_140342723849664(2.681782) q[1],q[2];
ryy_140342723846976(-0.68840069) q[2],q[3];
h q[0];
h q[1];
h q[2];
h q[3];
ry(0.59281313) q[0];
ry(0.37115771) q[1];
ry(-1.5105467) q[2];
ry(-0.6389901) q[3];
ryy_140342723819984(0.22002716) q[0],q[1];
ryy_140342723817728(-0.56065106) q[1],q[2];
ryy_140342723818400(0.96522439) q[2],q[3];
rx(-0.3396121) q[0];
rx(-1.727465) q[1];
rx(-0.42159155) q[2];
rx(1.7421111) q[3];
rz(1.4653542) q[0];
rz(0.40973204) q[1];
rz(-1.4750557) q[2];
rz(0.57165903) q[3];
crz(1.8032013) q[0],q[1];
crz(-0.56331378) q[0],q[2];
crz(2.9468753) q[0],q[3];
crz(0.98638278) q[1],q[0];
crz(-2.6939266) q[1],q[2];
crz(0.63895011) q[1],q[3];
crz(0.59464163) q[2],q[0];
crz(-2.2344394) q[2],q[1];
crz(-1.2533925) q[2],q[3];
crz(3.1607244) q[3],q[0];
crz(1.0943882) q[3],q[1];
crz(0.4885228) q[3],q[2];
rx(-1.0675564) q[0];
rx(3.3055) q[1];
rx(0.66384506) q[2];
rx(1.4313847) q[3];
rz(-0.48510855) q[0];
rz(-0.74189115) q[1];
rz(-1.8911432) q[2];
rz(1.7016194) q[3];
rx(2.0740814) q[0];
rx(1.047114) q[1];
rx(2.112663) q[2];
rx(2.9151456) q[3];
rz(-1.9631166) q[0];
rz(-2.0791314) q[1];
rz(-0.076765172) q[2];
rz(1.0376211) q[3];
crz(1.8123597) q[0],q[1];
crz(1.7410182) q[0],q[2];
crz(0.41524076) q[0],q[3];
crz(0.93953556) q[1],q[0];
crz(3.7293663) q[1],q[2];
crz(2.1310484) q[1],q[3];
crz(-1.3359119) q[2],q[0];
crz(-0.23219846) q[2],q[1];
crz(-0.55000538) q[2],q[3];
crz(0.57471168) q[3],q[0];
crz(3.159843) q[3],q[1];
crz(-2.4017704) q[3],q[2];
rx(0.61463392) q[0];
rx(-1.4282603) q[1];
rx(-1.304693) q[2];
rx(-2.5509853) q[3];
rz(-0.3020356) q[0];
rz(-1.9636184) q[1];
rz(0.99815762) q[2];
rz(-1.7005812) q[3];
rx(-0.64024317) q[0];
rx(-0.2007807) q[1];
rx(0.90376776) q[2];
rx(2.3851562) q[3];
rz(2.4127071) q[0];
rz(-0.28417486) q[1];
rz(-1.7011189) q[2];
rz(-0.92116088) q[3];
crz(0.0017415757) q[0],q[1];
crz(-0.47385937) q[0],q[2];
crz(2.387939) q[0],q[3];
crz(-2.3050964) q[1],q[0];
crz(-0.23158967) q[1],q[2];
crz(2.1182759) q[1],q[3];
crz(0.20440125) q[2],q[0];
crz(-0.68149328) q[2],q[1];
crz(-0.64930302) q[2],q[3];
crz(-0.093186758) q[3],q[0];
crz(2.0264144) q[3],q[1];
crz(-1.1651168) q[3],q[2];
rx(0.37997538) q[0];
rx(-1.5747147) q[1];
rx(-2.8667285) q[2];
rx(2.0151579) q[3];
rz(0.00032254076) q[0];
rz(-0.29120344) q[1];
rz(1.9664079e-09) q[2];
rz(-0.0015134607) q[3];
measure q[0] -> c[0];
measure q[1] -> c[1];
measure q[2] -> c[2];
measure q[3] -> c[3];