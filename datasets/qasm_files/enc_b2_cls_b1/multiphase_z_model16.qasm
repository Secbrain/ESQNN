OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
creg c[4];
rz(1.408458) q[0];
rz(-2.2690859) q[1];
rz(0.39256006) q[2];
rz(-0.42664155) q[3];
rx(0.36491603) q[0];
rx(0.80718172) q[1];
rx(-1.5602379) q[2];
rx(-0.055706803) q[3];
ry(-0.71252787) q[0];
ry(0.26294982) q[1];
ry(1.1325269) q[2];
ry(0.88280946) q[3];
rz(0.97636819) q[0];
rz(0.35255447) q[1];
rz(-0.5103265) q[2];
rz(0.048219867) q[3];
rz(0.77882069) q[0];
rz(-0.084266245) q[1];
rz(0.84605557) q[2];
rz(1.1856203) q[3];
rx(-0.44653532) q[0];
rx(0.80450153) q[1];
rx(0.83081746) q[2];
rx(-0.41165313) q[3];
ry(-0.19528379) q[0];
ry(1.0318246) q[1];
ry(-0.64199561) q[2];
ry(-1.0687633) q[3];
rz(-0.03808498) q[0];
rz(-0.83229762) q[1];
rz(0.8178792) q[2];
rz(0.14809109) q[3];
rx(0.45274001) q[0];
rx(2.4146528) q[1];
rx(-1.4880781) q[2];
rx(4.3223372) q[3];
rz(0.077660345) q[0];
rz(0.73295808) q[1];
rz(0.0021119332) q[2];
rz(-0.13025495) q[3];
crz(0.77406126) q[0],q[1];
crz(-0.60556006) q[2],q[3];
crz(9.4552568e-05) q[1],q[2];
measure q[0] -> c[0];
measure q[1] -> c[1];
measure q[2] -> c[2];
measure q[3] -> c[3];