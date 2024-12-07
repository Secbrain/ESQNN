OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
creg c[4];
rx(-0.67444628) q[0];
rx(-1.8892931) q[1];
rx(-1.8424436) q[2];
rx(0.13227719) q[3];
ry(-0.79287028) q[0];
ry(1.2297094) q[1];
ry(0.077734008) q[2];
ry(1.8036107) q[3];
rz(-0.33883819) q[0];
rz(-0.46696821) q[1];
rz(-0.40187645) q[2];
rz(-1.3109723) q[3];
rx(0.03079219) q[0];
rx(-0.59218955) q[1];
rx(-1.1771181) q[2];
rx(1.740944) q[3];
rx(-0.29608187) q[0];
rx(-0.34736946) q[1];
rx(-0.49671268) q[2];
rx(-1.3010066) q[3];
ry(1.3098557) q[0];
ry(-0.26663041) q[1];
ry(0.19697873) q[2];
ry(-0.69921434) q[3];
rz(1.1395644) q[0];
rz(0.19117494) q[1];
rz(-0.0094624413) q[2];
rz(0.35460788) q[3];
rx(-0.42382941) q[0];
rx(1.0711756) q[1];
rx(2.7124791) q[2];
rx(-0.19352838) q[3];
rx(-0.21597484) q[0];
rx(-0.4064627) q[1];
rx(1.8196173) q[2];
rx(-1.2796215) q[3];
rz(2.7516101e-10) q[0];
rz(-0.75145018) q[1];
rz(-0.13943192) q[2];
rz(0.72749776) q[3];
crz(0.094045855) q[0],q[1];
crz(-8.8302898e-10) q[1],q[2];
crz(-0.060910434) q[2],q[3];
measure q[0] -> c[0];
measure q[1] -> c[1];
measure q[2] -> c[2];
measure q[3] -> c[3];
