OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
creg c[4];
ry(-0.17038141) q[0];
ry(-0.30279297) q[1];
ry(-1.2868071) q[2];
ry(-1.3662828) q[3];
rz(-0.046252239) q[0];
rz(-0.61495847) q[1];
rz(1.2366945) q[2];
rz(-0.81435615) q[3];
rx(1.14621) q[0];
rx(-1.1787332) q[1];
rx(-0.03667279) q[2];
rx(0.67180979) q[3];
ry(0.92422974) q[0];
ry(0.2697157) q[1];
ry(0.62853712) q[2];
ry(-0.70661885) q[3];
ry(-3.8273087) q[0];
ry(2.7792518) q[1];
ry(-2.8362236) q[2];
ry(-0.21359889) q[3];
rz(-1.4262751) q[0];
rz(1.393379) q[1];
rz(1.613754) q[2];
rz(1.5397466) q[3];
cz q[0],q[1];
cz q[2],q[3];
ry(1.1991733) q[1];
ry(3.5754728) q[2];
rz(0.9403103) q[1];
rz(2.3972716) q[2];
cz q[1],q[2];
ry(2.7547772) q[0];
ry(-1.5488096) q[1];
ry(0.096309803) q[2];
ry(-2.2287977) q[3];
rz(1.7391379e-10) q[0];
rz(0.41230148) q[1];
rz(0.13903558) q[2];
rz(-0.71230781) q[3];
cz q[0],q[1];
cz q[2],q[3];
ry(-1.7358797) q[1];
ry(2.9221609) q[2];
rz(-0.57108831) q[1];
rz(0.66785687) q[2];
cz q[1],q[2];
measure q[0] -> c[0];
measure q[1] -> c[1];
measure q[2] -> c[2];
measure q[3] -> c[3];