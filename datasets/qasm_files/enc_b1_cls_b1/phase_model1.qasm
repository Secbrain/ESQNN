OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
creg c[4];
rx(1.9269153) q[0];
rx(1.4872841) q[1];
rx(0.9007172) q[2];
rx(-2.105521) q[3];
rx(0.67841846) q[0];
rx(-1.2345449) q[1];
rx(-0.043067478) q[2];
rx(-1.6046669) q[3];
rx(-0.75213528) q[0];
rx(1.648723) q[1];
rx(-0.39247864) q[2];
rx(-1.4036071) q[3];
rx(-0.72788131) q[0];
rx(-0.55943018) q[1];
rx(-0.76883888) q[2];
rx(0.76244539) q[3];
rx(-0.021042343) q[0];
rx(-0.23475575) q[1];
rx(0.865987) q[2];
rx(-1.3300314) q[3];
rz(-0.0054390295) q[0];
rz(-0.056760807) q[1];
rz(-2.4797405e-08) q[2];
rz(-0.898238) q[3];
measure q[0] -> c[0];
measure q[1] -> c[1];
measure q[2] -> c[2];
measure q[3] -> c[3];
