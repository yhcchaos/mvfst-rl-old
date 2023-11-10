mm-tunnelserver --ingress-log=mahi_tunnel_test/server_ingress.log --egress-log=mahi_tunnel_test/server_egress.log

mm-delay 30 mm-link train/traces/12mbps.trace train/traces/12mbps.trace --uplink-log=mahi_tunnel_test/data_uplink.log --downlink-log=mahi_tunnel_test/ack_downlink.log mm-loss uplink 0.01
mm-tunnelclient $MAHIMAHI_BASE 50326 100.64.0.4 100.64.0.3 --ingress-log=mahi_tunnel_test/client_tunnel_ingress.log --egress-log=mahi_tunnel_test/client_tunnel_egress.log