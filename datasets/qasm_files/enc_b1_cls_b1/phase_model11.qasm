OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
creg c[4];
rx(0.433438) q[0];
rx(-0.71719056) q[1];
rx(1.0553693) q[2];
rx(-1.4533969) q[3];
rx(0.46515071) q[0];
rx(0.37139151) q[1];
rx(-0.0046567856) q[2];
rx(0.079549439) q[3];
rx(0.37817848) q[0];
rx(0.70511413) q[1];
rx(-1.7236974) q[2];
rx(-0.84348106) q[3];
rx(0.43514356) q[0];
rx(0.2658872) q[1];
rx(-0.58709854) q[2];
rx(0.082688846) q[3];
ry(3.9868442e-09) q[0];
ry(-1.5678422) q[1];
ry(-0.40771228) q[2];
ry(2.5526726) q[3];
rz(0.00010926718) q[0];
rz(-0.08648926) q[1];
rz(0.0094632879) q[2];
rz(-1.2486902) q[3];
cx q[0],q[1];
cx q[2],q[3];
ry(1.5678649) q[1];
ry(-0.63056707) q[2];
rz(-0.081365682) q[1];
rz(-0.0017839001) q[2];
cx q[1],q[2];
measure q[0] -> c[0];
measure q[1] -> c[1];
measure q[2] -> c[2];
measure q[3] -> c[3];
