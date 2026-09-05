"""Four-switch diamond topology used by the SFT reroute experiment."""

from mininet.topo import Topo


class DiamondTopo(Topo):
    def build(self):
        s1 = self.addSwitch("s1", dpid="0000000000000001")
        s2 = self.addSwitch("s2", dpid="0000000000000002")
        s3 = self.addSwitch("s3", dpid="0000000000000003")
        s4 = self.addSwitch("s4", dpid="0000000000000004")

        h1 = self.addHost("h1", ip="10.0.0.1/24")
        h2 = self.addHost("h2", ip="10.0.0.2/24")
        h3 = self.addHost("h3", ip="10.0.0.3/24")

        # Port order is intentional and is part of the test fixture:
        # s1: s2=1, s3=2, h1=3
        # s2: s1=1, s4=2
        # s3: s1=1, s4=2
        # s4: s2=1, s3=2, h2=3, h3=4
        self.addLink(s1, s2)
        self.addLink(s1, s3)
        self.addLink(h1, s1)
        self.addLink(s2, s4)
        self.addLink(s3, s4)
        self.addLink(h2, s4)
        self.addLink(h3, s4)


topos = {"diamond": lambda: DiamondTopo()}
