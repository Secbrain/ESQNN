OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
creg c[4];
h q[0];
h q[1];
h q[2];
h q[3];
rz(0.67780983) q[0];
rz(-0.0017525852) q[1];
rz(-1.2441523) q[2];
rz(1.4180372) q[3];
rzz(-0.0011879195) q[0],q[1];
rzz(0.002180483) q[1],q[2];
rzz(-1.7642542) q[2],q[3];
h q[0];
h q[1];
h q[2];
h q[3];
rz(-0.83106446) q[0];
rz(1.4087291) q[1];
rz(0.53136903) q[2];
rz(1.4497354) q[3];
rzz(-1.1707447) q[0],q[1];
rzz(0.748555) q[1],q[2];
rzz(0.7703445) q[2],q[3];
h q[0];
h q[1];
h q[2];
h q[3];
rz(-0.84339315) q[0];
rz(-0.43651173) q[1];
rz(0.38062906) q[2];
rz(-0.17690393) q[3];
rzz(0.36815101) q[0],q[1];
rzz(-0.16614905) q[1],q[2];
rzz(-0.067334779) q[2],q[3];
h q[0];
h q[1];
h q[2];
h q[3];
rz(0.23560905) q[0];
rz(0.49744993) q[1];
rz(-0.64722449) q[2];
rz(-0.32472992) q[3];
rzz(0.11720371) q[0],q[1];
rzz(-0.32196179) q[1],q[2];
rzz(0.21017316) q[2],q[3];
h q[0];
h q[1];
h q[2];
h q[3];
rz(-1.0479207) q[0];
rz(2.2054734) q[1];
rz(-1.6851012) q[2];
rz(-0.52296686) q[3];
rzz(-2.3111613) q[0],q[1];
rzz(-3.7164459) q[1],q[2];
rzz(0.88125205) q[2],q[3];
h q[0];
h q[1];
h q[2];
h q[3];
rz(-0.69303006) q[0];
rz(-0.14844112) q[1];
rz(0.52521294) q[2];
rz(-0.53288358) q[3];
rzz(0.10287416) q[0],q[1];
rzz(-0.077963196) q[1],q[2];
rzz(-0.27987736) q[2],q[3];
h q[0];
h q[1];
h q[2];
h q[3];
rz(-1.2276394) q[0];
rz(0.83865774) q[1];
rz(-0.31661475) q[2];
rz(-1.4703326) q[3];
rzz(-1.0295693) q[0],q[1];
rzz(-0.26553142) q[1],q[2];
rzz(0.46552899) q[2],q[3];
h q[0];
h q[1];
h q[2];
h q[3];
rz(1.6236553) q[0];
rz(0.47365865) q[1];
rz(1.0424601) q[2];
rz(-0.62522483) q[3];
rzz(0.76905841) q[0],q[1];
rzz(0.49377024) q[1],q[2];
rzz(-0.6517719) q[2],q[3];
rx(3.0924404) q[0];
rx(0.36141238) q[1];
rx(0.9132399) q[2];
rx(0.14206304) q[3];
rz(-0.81439137) q[0];
rz(0.0017517862) q[1];
rz(0.14892377) q[2];
rz(-0.0051993132) q[3];
cx q[0],q[1];
cx q[1],q[2];
cx q[2],q[3];
measure q[0] -> c[0];
measure q[1] -> c[1];
measure q[2] -> c[2];
measure q[3] -> c[3];
