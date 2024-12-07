OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
creg c[4];
rx(0.68387526) q[0];
rx(-1.3245894) q[1];
rx(-0.51608175) q[2];
rx(0.60018426) q[3];
rx(-0.47022083) q[0];
rx(-0.60864353) q[1];
rx(-0.046192024) q[2];
rx(-1.6457398) q[3];
rx(-0.48332742) q[0];
rx(-0.74029434) q[1];
rx(0.31428108) q[2];
rx(0.14155531) q[3];
rx(1.0348177) q[0];
rx(-0.62643778) q[1];
rx(-0.51509225) q[2];
rx(0.69028997) q[3];
ry(2.925282) q[0];
ry(0.39247903) q[1];
ry(-0.93084103) q[2];
ry(0.32640737) q[3];
crx(-1.1382756) q[0],q[3];
crx(0.69647068) q[1],q[0];
crx(-0.34964815) q[2],q[1];
crx(0.076227009) q[3],q[2];
ry(-1.5221533) q[0];
ry(1.0291356) q[1];
ry(0.67475152) q[2];
ry(2.9208269) q[3];
crx(1.5853134) q[0],q[1];
crx(-3.0936842) q[3],q[0];
crx(-0.88319993) q[2],q[3];
crx(-0.36241606) q[1],q[2];
ry(-3.297879) q[0];
ry(-0.45611492) q[1];
ry(0.23721276) q[2];
ry(0.55510688) q[3];
crx(-0.1414316) q[0],q[3];
crx(-1.8521354) q[1],q[0];
crx(-1.1936312) q[2],q[1];
crx(0.33175895) q[3],q[2];
ry(2.2857378) q[0];
ry(-0.1822874) q[1];
ry(0.12079185) q[2];
ry(1.0215451) q[3];
crx(1.7094576) q[0],q[1];
crx(0.16071236) q[3],q[0];
crx(-0.33435389) q[2],q[3];
crx(-3.1384251) q[1],q[2];
measure q[0] -> c[0];
measure q[1] -> c[1];
measure q[2] -> c[2];
measure q[3] -> c[3];
