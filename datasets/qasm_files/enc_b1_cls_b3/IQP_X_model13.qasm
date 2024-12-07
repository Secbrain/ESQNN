OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
creg c[4];
h q[0];
h q[1];
h q[2];
h q[3];
rx(0.52973324) q[0];
rx(-0.37780449) q[1];
rx(1.4173911) q[2];
rx(0.49107137) q[3];
rxx(-0.20013559) q[0],q[1];
rxx(-0.53549671) q[1],q[2];
rxx(0.69604015) q[2],q[3];
h q[0];
h q[1];
h q[2];
h q[3];
rx(0.91596478) q[0];
rx(0.17350815) q[1];
rx(0.18968929) q[2];
rx(-0.15840918) q[3];
rxx(0.15892735) q[0],q[1];
rxx(0.032912638) q[1],q[2];
rxx(-0.030048525) q[2],q[3];
h q[0];
h q[1];
h q[2];
h q[3];
rx(-1.9691961) q[0];
rx(-0.24453369) q[1];
rx(-0.67767137) q[2];
rx(0.47822806) q[3];
rxx(0.48153478) q[0],q[1];
rxx(0.16571347) q[1],q[2];
rxx(-0.32408148) q[2],q[3];
h q[0];
h q[1];
h q[2];
h q[3];
rx(-1.890582) q[0];
rx(0.29901153) q[1];
rx(1.5097411) q[2];
rx(-0.40140989) q[3];
rxx(-0.56530583) q[0],q[1];
rxx(0.45142999) q[1],q[2];
rxx(-0.60602498) q[2],q[3];
ry(-0.40712434) q[0];
ry(-2.1225526) q[1];
ry(-1.8747593) q[2];
ry(0.50110716) q[3];
crz(-3.4337394) q[0],q[3];
crz(0.89877367) q[1],q[0];
crz(-1.3752899) q[2],q[1];
crz(1.6709743) q[3],q[2];
ry(-3.1749747) q[0];
ry(1.7528607) q[1];
ry(0.82593662) q[2];
ry(-0.76005763) q[3];
crz(0.88298965) q[0],q[1];
crz(1.1559713) q[3],q[0];
crz(-0.33335108) q[2],q[3];
crz(3.4668236) q[1],q[2];
ry(-2.6337061) q[0];
ry(1.0891891) q[1];
ry(0.69835663) q[2];
ry(1.4587985) q[3];
crz(2.1683965) q[0],q[3];
crz(0.92217726) q[1],q[0];
crz(-3.4803736) q[2],q[1];
crz(2.2489572) q[3],q[2];
ry(1.7105137) q[0];
ry(0.065746851) q[1];
ry(-2.6694357) q[2];
ry(-0.08672104) q[3];
crz(-3.1766253) q[0],q[1];
crz(-1.6050541) q[3],q[0];
crz(1.8239157) q[2],q[3];
crz(3.0119724) q[1],q[2];
ry(-0.74332798) q[0];
ry(0.18036148) q[1];
ry(-2.3412299) q[2];
ry(0.68634164) q[3];
crz(3.8946016) q[0],q[3];
crz(-3.1726329) q[1],q[0];
crz(1.9527144) q[2],q[1];
crz(3.6213975) q[3],q[2];
ry(1.5435059) q[0];
ry(1.5435514) q[1];
ry(-2.6159694) q[2];
ry(-0.13926207) q[3];
crz(-0.38613096) q[0],q[1];
crz(0.88716227) q[3],q[0];
crz(0.023362326) q[2],q[3];
crz(-0.88338608) q[1],q[2];
measure q[0] -> c[0];
measure q[1] -> c[1];
measure q[2] -> c[2];
measure q[3] -> c[3];
