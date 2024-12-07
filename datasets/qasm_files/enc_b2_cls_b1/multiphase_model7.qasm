OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
creg c[4];
rx(1.8312736) q[0];
rx(-0.61592156) q[1];
rx(-0.60728222) q[2];
rx(-2.0597374) q[3];
ry(1.5289141) q[0];
ry(0.33786839) q[1];
ry(0.19153585) q[2];
ry(0.16352268) q[3];
rz(0.67101675) q[0];
rz(-0.40960962) q[1];
rz(-0.5302254) q[2];
rz(0.25328615) q[3];
rx(-0.199) q[0];
rx(0.61014169) q[1];
rx(-1.4391361) q[2];
rx(1.6620673) q[3];
rx(0.3555752) q[0];
rx(-1.8119957) q[1];
rx(0.46456537) q[2];
rx(-0.54800504) q[3];
ry(-1.059624) q[0];
ry(0.17400648) q[1];
ry(0.38216051) q[2];
ry(-0.19578159) q[3];
rz(-0.15132363) q[0];
rz(0.62557799) q[1];
rz(-0.62190396) q[2];
rz(-1.0873214) q[3];
rx(-1.3252078) q[0];
rx(0.37722504) q[1];
rx(-0.058415074) q[2];
rx(-1.4766152) q[3];
rx(-3.2408373) q[0];
rx(-1.0949183) q[1];
rx(1.8915536) q[2];
rx(0.023762064) q[3];
rz(-2.7505686) q[0];
rz(0.21588936) q[1];
rz(-1.8555182) q[2];
rz(2.6319101) q[3];
crz(-3.4596622) q[0],q[1];
crz(1.8081483) q[2],q[3];
rx(-2.6890354) q[0];
rx(-1.0404319) q[1];
rx(0.0015076136) q[2];
rx(1.4345231) q[3];
rz(-0.74972063) q[0];
rz(0.16970922) q[1];
rz(0.077493839) q[2];
rz(-0.31817976) q[3];
crz(0.27050287) q[1],q[2];
measure q[0] -> c[0];
measure q[1] -> c[1];
measure q[2] -> c[2];
measure q[3] -> c[3];